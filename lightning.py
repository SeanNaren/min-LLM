import logging
import math
import os
from typing import Union

import deepspeed
import fire
import pytorch_lightning as pl
import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm
from deepspeed.ops.adam import FusedAdam
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from xformers.components import LayerNormStyle
from xformers.triton import FusedLinear

from memory import CUDAMemoryCallback
from model import EncoderBlock, ModuleWrapperIgnores2ndArg
from profiler import GPTFLOPsEstimate


class LLM(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        learning_rate=6e-4,
        n_embd=512,
        block_size=128,
        n_layer=8,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        warmup_tokens=20,
        final_tokens=1000,
        sparse_block_size=128,
    ):
        super().__init__()

        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        blocks = self.hparams.block_size // self.hparams.sparse_block_size
        layout = torch.tril(
            torch.ones([self.hparams.n_head, blocks, blocks], dtype=torch.bool)
        )

        self.layout = layout
        self.block_size = self.hparams.block_size
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._tokens_seen = 0

    def configure_sharded_model(self) -> None:
        blocks = []
        for i in range(self.hparams.n_layer):
            pose_encoding = i == 0
            last_encoder_block = i == (self.hparams.n_layer - 1)
            encoder = EncoderBlock(
                pose_encoding=pose_encoding,
                causal=True,
                use_rotary_embeddings=True,
                last_encoder_block=last_encoder_block,
                layout=self.layout,
                dim_model=self.hparams.n_embd,
                num_heads=self.hparams.n_head,
                seq_len=self.hparams.block_size,
                sparse_block_size=self.hparams.sparse_block_size,
                vocab_size=self.hparams.vocab_size,
                attn_pdrop=self.hparams.attn_pdrop,
                mlp_pdrop=self.hparams.mlp_pdrop,
                residual_pdrop=self.hparams.resid_pdrop,
                layer_norm_style=LayerNormStyle.Pre,
                hidden_layer_multiplier=self.hparams.hidden_layer_multiplier,
            )
            blocks.append(ModuleWrapperIgnores2ndArg(encoder))

        self.encoders = torch.nn.ModuleList(blocks)

        # decoder head
        self.ln_f = FusedLayerNorm(self.hparams.n_embd)
        self.head = FusedLinear(
            self.hparams.n_embd, self.hparams.vocab_size, bias=False
        )

        # todo: when using model parallelism, this will be wrong. will need to redo
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (FusedLinear, nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self._tokens_seen = 0

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self):
        # Create the optimizer and the training schedule:
        # - Handle the per-param weight decay
        no_decay = ["bias", "LayerNorm.weight", "FusedLayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # - Start with a warm up, ramp up then cosine
        optimizer = FusedAdam(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )

        def update_lr(*_):
            config = self.hparams

            if self._tokens_seen < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, config.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[update_lr, update_lr],
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        # encode to latent space
        for encoder in self.encoders:
            x = deepspeed.checkpointing.checkpoint(encoder, x, self.dummy_tensor)

        # translate the predictions into tokens
        prediction = self.ln_f(x)
        logits = self.head(prediction)

        return logits

    def training_step(self, batch, _):
        src, targets = batch

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits = self(src)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=self.trainer.global_step,
        )

        return loss


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[
            None, ...
        ].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])


def main(
    batch_size_per_gpu: int = 36,
    accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    epochs: int = 1,
    block_size: int = 2048,
    warmup: int = 20,
    devices: int = 1,
    precision: Union[str, int] = 16,
    n_layer: int = 24,
    n_head: int = 24,
    n_embd: int = 2304,
    attention: str = "scaled_dot_product",
    sparse_block_size: int = 128,
    strategy: str = "deepspeed",
    stage: int = 3,
):
    seed_everything(42)

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size)
    global_batch_size = batch_size_per_gpu * devices
    config = {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "reduce_bucket_size": n_embd * n_embd,
            "stage3_prefetch_bucket_size": 0.9 * n_embd * n_embd,
            "stage3_param_persistence_threshold": 10 * n_embd,
        },
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "bf16": {"enabled": precision == "bf16"},
        "fp16": {"enabled": precision == 16},
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = LLM(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        attention=attention,
        sparse_block_size=sparse_block_size,
        warmup_tokens=global_batch_size * warmup,
        final_tokens=epochs * len(train_dataset) * block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=DeepSpeedStrategy(config=config, logging_level=logging.INFO)
        if strategy == "deepspeed"
        else strategy,
        callbacks=[
            GPTFLOPsEstimate(
                global_batch_size=global_batch_size,
                hidden_size=n_embd,
                n_layer=n_layer,
                block_size=block_size,
                vocab_size=train_dataset.vocab_size,
                activation_checkpointing=True,
            ),
            CUDAMemoryCallback(),
        ],
        limit_train_batches=50,
        max_epochs=epochs,
        precision=precision,
        gradient_clip_val=1,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    fire.Fire(main)
