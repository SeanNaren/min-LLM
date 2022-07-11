import argparse
import logging
import os
import random
import time
from pprint import pprint
from typing import Union

import deepspeed
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model import LLM


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


class Profiler:
    def __init__(
        self,
        global_batch_size: int,
        hidden_size: int,
        num_devices: int,
        device: torch.device,
        n_layer: int,
        block_size: int,
        vocab_size: int,
        activation_checkpointing: bool = False,
    ):
        self.global_batch_size = global_batch_size
        self.activation_checkpointing = activation_checkpointing
        self.num_devices = num_devices
        self.device = device
        self.s = block_size
        self.h = hidden_size
        self.l = n_layer
        self.v = vocab_size

        self.num_parameters = (
            self.l * (12 * self.h ** 2 + 13 * self.h)
            + self.v * self.h
            + self.s * self.h
            + 2 * self.h
        ) / 10 ** 9
        print(f"Number of parameters: {self.num_parameters:.2f} Billion")

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

    def start_profiling(self):
        torch.cuda.synchronize(self.device)
        self.start = time.time()

    def end_profiling(self, num_steps) -> None:
        torch.cuda.synchronize(self.device)
        total_time = time.time() - self.start
        factor = 4 if self.activation_checkpointing else 3
        per_iteration_time = total_time / num_steps
        # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
        # https://arxiv.org/pdf/2104.04473.pdf).
        # https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251/files
        flops_per_iteration = (
            24 * factor * self.global_batch_size * self.s * self.l * (self.h ** 2)
        ) * (1.0 + (self.s / (6.0 * self.h)) + (self.v / (16.0 * self.l * self.h)))
        tflops = flops_per_iteration / (
            per_iteration_time * self.num_devices * (10 ** 12)
        )
        print(
            f"Estimates: {tflops:.2f}TFLOPs Avg Iteration Time: {per_iteration_time:.2f}s"
        )
        torch.cuda.synchronize(self.device)
        max_memory = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        print(f"Average Peak CUDA memory {max_memory:.2f} MiB")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(
    batch_size_per_gpu: int = 36,
    num_workers: int = 4,
    epochs: int = 1,
    block_size: int = 2048,
    warmup: int = 20,
    profile_start_step: int = 20,
    profile_num_steps: int = 20,
    logging_level: int = logging.INFO,
    n_layer: int = 24,
    n_head: int = 24,
    n_embd: int = 2304,
    sparse_block_size: int = 128,
    stage: int = 3,
    local_rank: int = 0,
    precision: Union[str, int] = "bf16",
):
    seed_everything(42)
    deepspeed.init_distributed()
    deepspeed.utils.logging.logger.setLevel(logging_level)
    local_rank = int(local_rank)
    local_rank_zero = local_rank == 0
    root_device = torch.device(f"cuda:{local_rank}")
    num_devices = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(root_device)

    # todo: replace with a meaningful dataset for us to train on
    if not os.path.exists("input.txt") and local_rank_zero:
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
    torch.distributed.barrier()  # wait for main process to download input.txt

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
    model = LLM(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
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
        "gradient_clipping": 1,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "bf16": {"enabled": precision == "bf16"},
        "fp16": {"enabled": precision == 16},
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
            activation_checkpointing=True,
            num_devices=num_devices,
            device=root_device,
        )

    data_length = len(train_loader)

    for x, batch in enumerate(train_loader):
        if x == profile_start_step and local_rank_zero:
            prof.start_profiling()
        src, targets = batch
        batch = (src.to(root_device, non_blocking=True), targets.to(root_device, non_blocking=True))
        loss = deepspeed_engine(batch)
        if local_rank_zero:
            pprint(f"[{x}/{data_length}]: Loss: {loss}")
        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        if x == profile_start_step + profile_num_steps and local_rank_zero:
            prof.end_profiling(profile_num_steps)
            break


if __name__ == "__main__":
    fire.Fire(main)
