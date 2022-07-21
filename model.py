import math
from typing import List, Optional, Tuple

import deepspeed
import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm
from deepspeed.ops.adam import FusedAdam
from torch.nn import functional as F
from xformers.components import (Activation, ResidualNormStyle, MultiHeadDispatch,
                                 RequiresWrappedInputs, Residual)
from xformers.components.attention import BlockSparseAttention
from xformers.components.feedforward import MLP
from xformers.components.positional_embedding import VocabEmbedding
from xformers.triton import FusedLinear


class ModuleWrapperIgnores2ndArg(nn.Module):
    """
    Taken from https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11.
    When using activation checkpointing, we have to ensure that there exists an input that requires gradients,
    otherwise this breaks the autograd tape.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class EncoderLayerNorm(nn.Module, RequiresWrappedInputs):
    def __init__(
        self, module: nn.Module, dim_model: int, residual_norm_style: ResidualNormStyle
    ):
        super().__init__()
        self.sublayer = module
        self.residual_norm_style = residual_norm_style
        self.norm = FusedLayerNorm(dim_model)
        self.wrap_inputs = isinstance(self.sublayer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        x = inputs
        if self.residual_norm_style == ResidualNormStyle.Pre:
            x = self._apply_xformer_pre_norm(x)
        if self.wrap_inputs:
            x = self.sublayer(inputs=x, **kwargs)
        else:
            x = self.sublayer(*x, **kwargs)
        if self.residual_norm_style == ResidualNormStyle.Post:
            x = self.norm(x)
        return x

    def _apply_xformer_pre_norm(self, inputs):
        # Perf improvement: if the inputs are all the same, only norm once
        ids = [id(x) for x in inputs]
        if ids.count(ids[0]) == len(ids):
            # The same tensor is passed multiple times
            x_norm = self.norm(inputs[0])
            inputs_normed = [x_norm for _ in inputs]
        else:
            # The inputs differ, norm them all
            inputs_normed = [self.norm(x_) for x_ in inputs]
        return inputs_normed


class EncoderBlock(nn.Module):
    def __init__(
        self,
        pose_encoding: bool,
        dim_model: int,
        num_heads: int,
        seq_len: int,
        vocab_size: int,
        sparse_block_size: int,
        causal: bool,
        use_rotary_embeddings: bool,
        layout: torch.tensor,
        mlp_pdrop: float,
        residual_pdrop: float,
        attn_pdrop: float,
        residual_norm_style: ResidualNormStyle,
        hidden_layer_multiplier: int,
        last_encoder_block: bool,
    ):
        super().__init__()
        self.pose_encoding = None
        if pose_encoding:
            self.pose_encoding = VocabEmbedding(
                dim_model=dim_model, seq_len=seq_len, vocab_size=vocab_size
            )
        attention = BlockSparseAttention(
            layout=layout,
            block_size=sparse_block_size,
            dropout=attn_pdrop,
            causal=causal,
            num_heads=num_heads,
        )
        multi_head_attention = MultiHeadDispatch(
            dim_model=dim_model,
            num_heads=num_heads,
            attention=attention,
            use_rotary_embeddings=use_rotary_embeddings,
            residual_dropout=residual_pdrop,
        )
        self.attention = self._wrap_with_residual_layer_norm(
            multi_head_attention, dim_model=dim_model, residual_norm_style=residual_norm_style
        )

        ff = MLP(
            dim_model=dim_model,
            dropout=mlp_pdrop,
            activation=Activation.GeLU,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

        ff = self._wrap_with_residual_layer_norm(
            ff, dim_model=dim_model, residual_norm_style=residual_norm_style
        )
        if residual_norm_style == ResidualNormStyle.Pre and last_encoder_block:
            ff = EncoderLayerNorm(ff, dim_model, ResidualNormStyle.Post)
        self.ff = ff

    def _wrap_with_residual_layer_norm(
        self, module, dim_model, residual_norm_style
    ) -> Residual:
        return Residual(EncoderLayerNorm(module, dim_model, residual_norm_style))

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.pose_encoding is not None:
            x = self.pose_encoding(x)

        # Handle the optional input masking, differs on Q, K, V
        if input_mask is not None:
            q = x
            k = x * input_mask.unsqueeze(-1)
            v = k
        else:
            q, k, v = x, x, x

        # Pre/Post norms and residual paths are already handled
        x = self.attention(inputs=[q, k, v], att_mask=att_mask)
        x = self.ff(inputs=[x])
        return x


class LLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        weight_decay: float = 0.1,
        betas: Tuple = (0.9, 0.95),
        learning_rate: float = 6e-4,
        n_embd: int = 512,
        block_size: int = 128,
        n_layer: int = 8,
        n_head: int = 8,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        hidden_layer_multiplier: int = 4,
        warmup_tokens: int = 20,
        final_tokens: int = 1000,
        sparse_block_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight_decay = weight_decay
        self.betas = betas
        self.learning_rate = learning_rate
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.mlp_pdrop = mlp_pdrop
        self.sparse_block_size = sparse_block_size
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.block_size = self.block_size

        blocks = block_size // self.sparse_block_size
        self.layout = torch.tril(
            torch.ones([self.n_head, blocks, blocks], dtype=torch.bool)
        )

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.tokens_seen = 0

        blocks = []
        for i in range(self.n_layer):
            pose_encoding = i == 0
            last_encoder_block = i == (self.n_layer - 1)
            encoder = EncoderBlock(
                pose_encoding=pose_encoding,
                causal=True,
                use_rotary_embeddings=True,
                last_encoder_block=last_encoder_block,
                layout=self.layout,
                dim_model=self.n_embd,
                num_heads=self.n_head,
                seq_len=self.block_size,
                sparse_block_size=self.sparse_block_size,
                vocab_size=self.vocab_size,
                attn_pdrop=self.attn_pdrop,
                mlp_pdrop=self.mlp_pdrop,
                residual_pdrop=self.resid_pdrop,
                residual_norm_style=ResidualNormStyle.Pre,
                hidden_layer_multiplier=self.hidden_layer_multiplier,
            )
            blocks.append(ModuleWrapperIgnores2ndArg(encoder))

        self.encoders = torch.nn.ModuleList(blocks)

        # decoder head
        self.ln_f = FusedLayerNorm(self.n_embd)
        self.head = FusedLinear(self.n_embd, self.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (FusedLinear, nn.Linear, nn.Embedding)):
            # taken from
            # https://github.com/bigscience-workshop/Megatron-DeepSpeed/commit/a6bf1a042dd28eae77200461d735a399377fd4c3
            module.weight.data.normal_(mean=0.0, std=0.006)
            if isinstance(module, (nn.Linear, FusedLinear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, FusedLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self.tokens_seen = 0

    def configure_optimizers(self):
        # Create the optimizer and the training schedule:
        # - Handle the per-param weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # - Start with a warm up, ramp up then cosine
        optimizer = FusedAdam(optim_groups, lr=self.learning_rate, betas=self.betas)

        def update_lr(*_):

            if self.tokens_seen < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens_seen) / float(max(1, self.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self.tokens_seen - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return lr_mult

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[update_lr, update_lr],
        )
        return optimizer, lr_scheduler

    def forward(self, batch):
        x, targets = batch

        # Update the tokens we've seen (tracked for LR scheduling)
        self.tokens_seen += (x >= 0).numel()

        # encode to latent space
        for encoder in self.encoders:
            x = deepspeed.checkpointing.checkpoint(encoder, x, self.dummy_tensor)

        # translate the predictions into tokens
        prediction = self.ln_f(x)
        logits = self.head(prediction)

        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
