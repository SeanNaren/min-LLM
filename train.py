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
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from model import LLM


class CharDataset(IterableDataset):
    def __init__(self, block_size):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        ds = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.block_size = block_size

        block_size = block_size + 1

        def convert_to_features(examples):
            examples = self.tokenizer(examples['text'])
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        ds = ds.map(convert_to_features, remove_columns=['text', 'id'], batched=True)

        self.ds = ds

    def __iter__(self):
        self.ds.shuffle()
        for chunk in self.ds:
            chunk = chunk["input_ids"]
            # src and target are off by one, we want the model to predict the next word
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


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
    num_iterations: int = 5000,
    batch_size_per_gpu: int = 36,
    num_workers: int = 8,
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
    seed_everything(42 + local_rank)
    deepspeed.init_distributed()
    deepspeed.utils.logging.logger.setLevel(logging_level)
    local_rank = int(local_rank)
    local_rank_zero = local_rank == 0
    root_device = torch.device(f"cuda:{local_rank}")
    num_devices = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(root_device)

    train_dataset = CharDataset(block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
    )

    global_batch_size = batch_size_per_gpu * num_devices
    model = LLM(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        warmup_tokens=global_batch_size * warmup,
        final_tokens=num_iterations * block_size,
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

    train_loader = iter(train_loader)
    for x in range(num_iterations):
        batch = next(train_loader)
        if x == profile_start_step and local_rank_zero:
            prof.start_profiling()
        src, targets = batch
        batch = (src.to(root_device, non_blocking=True), targets.to(root_device, non_blocking=True))
        loss = deepspeed_engine(batch)
        if local_rank_zero:
            pprint(f"[{x}]: Loss: {loss}")
        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        if x == profile_start_step + profile_num_steps and local_rank_zero:
            prof.end_profiling(profile_num_steps)
            break


if __name__ == "__main__":
    fire.Fire(main)
