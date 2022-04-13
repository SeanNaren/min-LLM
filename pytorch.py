import argparse
import logging
import math
import os
from pprint import pprint

import deepspeed
import fire
import torch
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam
from deepspeed.profiling.flops_profiler import FlopsProfiler
from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from xformers.factory.model_factory import xFormer, xFormerConfig


class GPT(torch.nn.Module):

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
        self.attention = attention
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": self.n_layer,
                "dim_model": self.n_embd,
                "layer_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.block_size,
                    "vocab_size": self.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": self.n_head,
                    "residual_dropout": self.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": self.attention,
                        "dropout": self.attn_pdrop,
                        "causal": True,
                        "seq_len": self.block_size,
                        "num_rules": self.n_head,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",  # Use MLP if Triton is not available
                    "dropout": self.mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": self.hidden_layer_multiplier,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        self.model = xFormer.from_config(config)

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.block_size = self.block_size
        self.apply(self._init_weights)

        self._tokens_seen = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self._tokens_seen = 0

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
        optimizer = FusedAdam(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )

        def update_lr(*_):

            if self._tokens_seen < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, self.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return lr_mult

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[update_lr, update_lr],
        )
        return optimizer, lr_scheduler

    def forward(self, batch):
        src, targets = batch

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # predict the next tokens (in latent space)
        prediction = self.model(src)

        # translate the predictions into tokens
        prediction = self.ln_f(prediction)
        logits = self.head(prediction)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i: i + self.block_size + 1]
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
        accumulate_batch_size: int = 2,
        batch_size: int = 2,
        num_workers: int = 4,
        epochs: int = 1,
        block_size: int = 2048,
        warmup: int = 20,
        start_idx: int = 20,
        end_idx: int = 40,
        logging_level: int = logging.WARN,
        n_layer: int = 14,
        n_head: int = 16,
        n_embd: int = 2048
):
    seed_everything(42)
    deepspeed.init_distributed()
    deepspeed.utils.logging.logger.setLevel(logging_level)

    local_rank = int(os.environ['LOCAL_RANK'])
    local_rank_zero = local_rank == 0
    root_device = torch.device(f"cuda:{local_rank}")

    def print_fn(*args):
        if local_rank_zero:
            pprint(*args)

    print = print_fn

    torch.cuda.set_device(root_device)

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        attention="scaled_dot_product",
        warmup_tokens=accumulate_batch_size * warmup,
        final_tokens=epochs * len(train_dataset) * block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    optimizer, scheduler = model.configure_optimizers()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    config = {
        "gradient_accumulation_steps": accumulate_batch_size // batch_size,
        "fp16": {
            "enabled": True
        },
        "train_micro_batch_size_per_gpu": accumulate_batch_size // batch_size,
        "zero_allow_untested_optimizer": False,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        }
    }

    deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
        args=argparse.Namespace(device_rank=root_device.index),
        config=config,
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    if local_rank_zero:
        prof = FlopsProfiler(model)

    data_length = len(train_loader)

    for x, batch in enumerate(train_loader):
        if x == start_idx and local_rank_zero:
            prof.start_profile()
        src, targets = batch
        batch = (src.to(root_device), targets.to(root_device))
        loss = deepspeed_engine(batch)
        print(f"[{x}/{data_length}]: Loss: {loss}")
        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        if x == end_idx and local_rank_zero:
            prof.print_model_profile(x, detailed=False)
            prof.end_profile()
            break


if __name__ == '__main__':
    fire.Fire(main)
