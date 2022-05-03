import argparse
import logging
import math
import os
import time
from pprint import pprint
from typing import Union

import deepspeed
import fire
import torch
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam
from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from xformers.factory import xFormerEncoderConfig, xFormerEncoderBlock
from xformers.triton import FusedLayerNorm, FusedLinear


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
            sparse_block_size=128,
            local_rank=0,
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
        self.sparse_block_size = sparse_block_size
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens

        attention_kwargs = {
            "name": self.attention,
            "dropout": self.attn_pdrop,
            "causal": True,
            "seq_len": self.block_size,
            "num_rules": self.n_head,
        }
        if self.attention == "blocksparse":
            blocks = block_size // self.sparse_block_size
            layout = torch.tril(torch.ones([self.n_head, blocks, blocks], dtype=torch.bool))
            attention_kwargs["layout"] = layout
            attention_kwargs["block_size"] = self.sparse_block_size

        config = {
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
                "attention": attention_kwargs
            },
            "feedforward_config": {
                "name": "FusedMLP",
                "dropout": self.mlp_pdrop,
                "activation": "gelu",
                "hidden_layer_multiplier": self.hidden_layer_multiplier,
            },
        }

        self.config = xFormerEncoderConfig(**config)

        self.block_size = self.block_size

        self._tokens_seen = 0

        blocks = []
        for i in range(self.n_layer):
            # Label where this layer is in the stack
            # (for instance useful for the positional encoding, or late layer norm)
            if i > 0:
                self.config.layer_position.mark_not_first()
            if i < self.config.num_layers - 1:
                self.config.layer_position.mark_not_last()
            blocks.append(xFormerEncoderBlock.from_config(self.config))

        self.encoders = torch.nn.ModuleList(blocks)

        # decoder head
        self.ln_f = FusedLayerNorm(self.n_embd)
        self.head = FusedLinear(self.n_embd, self.vocab_size, bias=False)

        # todo: when using model parallelism, this may be wrong. will need to redo
        self.apply(self._init_weights)


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
        # encode to latent space
        encoders = self.encoders
        prediction = src.clone()
        for encoder in encoders:
            # prediction = deepspeed.checkpointing.checkpoint(encoder, prediction)
            prediction = encoder(prediction)

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


class Profiler:
    def __init__(self,
                 global_batch_size: int,
                 hidden_size: int,
                 num_devices: int,
                 device: torch.device,
                 n_layer: int,
                 block_size: int,
                 vocab_size: int,
                 activation_checkpointing: bool = False):
        self.global_batch_size = global_batch_size
        self.activation_checkpointing = activation_checkpointing
        self.num_devices = num_devices
        self.device = device
        h = hidden_size
        l = n_layer
        self.s = block_size
        v = vocab_size

        self.num_parameters = (l * (12 * h ** 2 + 13 * h) + v * h + self.s * h + 2 * h) / 10 ** 9
        print(f"Number of parameters: {self.num_parameters:.2f} Billion")

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

    def start_profiling(self):
        self.start = time.time()

    def end_profiling(self, num_steps) -> None:
        total_time = time.time() - self.start
        factor = 4 if self.activation_checkpointing else 3
        per_iteration_time = total_time / num_steps
        flops = self.num_parameters * factor * 2 * self.s * self.global_batch_size
        flops = flops / (per_iteration_time * self.num_devices * 1e3)
        print(f"Estimates: {flops:.2f}TFLOPs Avg Iteration Time: {per_iteration_time:.2f}s")

        torch.cuda.synchronize(self.device)
        max_memory = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        print(f"Average Peak CUDA memory {max_memory:.2f} MiB")


def main(
        batch_size_per_gpu: int = 2,
        accumulate_grad_batches: int = 1,
        num_workers: int = 4,
        epochs: int = 1,
        block_size: int = 2048,
        warmup: int = 20,
        profile_start_step: int = 20,
        profile_num_steps: int = 20,
        logging_level: int = logging.WARN,
        n_layer: int = 14,
        n_head: int = 16,
        n_embd: int = 2048,
        attention: str = "scaled_dot_product",
        sparse_block_size: int = 128,
        stage: int = 3,
        local_rank: int = 0,
        precision: Union[str, int] = 16,
):
    def print_fn(*args):
        if local_rank_zero:
            pprint(*args)

    print = print_fn

    seed_everything(42)
    deepspeed.init_distributed()
    deepspeed.utils.logging.logger.setLevel(logging_level)
    local_rank = int(local_rank)
    local_rank_zero = local_rank == 0
    root_device = torch.device(f"cuda:{local_rank}")
    num_devices = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(root_device)

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size)
    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
    )

    global_batch_size = batch_size_per_gpu * num_devices
    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        attention=attention,
        warmup_tokens=global_batch_size * warmup,
        final_tokens=epochs * len(train_dataset) * block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        sparse_block_size=sparse_block_size,
    )

    optimizer, scheduler = model.configure_optimizers()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    config = {
        "gradient_accumulation_steps": accumulate_grad_batches,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "zero_allow_untested_optimizer": False,
        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        },
        "bf16": {"enabled": precision == "bf16"},
        "fp16": {"enabled": precision == 16}
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
        prof = Profiler(
            global_batch_size=global_batch_size,
            hidden_size=n_embd,
            n_layer=n_layer,
            block_size=block_size,
            vocab_size=train_dataset.vocab_size,
            activation_checkpointing=False,
            num_devices=num_devices,
            device=root_device,
        )

    data_length = len(train_loader)

    for x, batch in enumerate(train_loader):
        if x == profile_start_step and local_rank_zero:
            prof.start_profiling()
        src, targets = batch
        batch = (src.to(root_device), targets.to(root_device))
        loss = deepspeed_engine(batch)
        print(f"[{x}/{data_length}]: Loss: {loss}")
        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        if x == profile_start_step + profile_num_steps and local_rank_zero:
            prof.end_profiling(profile_num_steps)
            break


if __name__ == '__main__':
    fire.Fire(main)
