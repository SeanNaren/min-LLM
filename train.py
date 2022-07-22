import argparse
import math
import os
import random
from typing import Optional, Union

import deepspeed
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

from data import CharDataset
from metrics import Metrics
from model import LLM
from profiler import Profiler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(
    num_iterations: int = 50000,
    batch_size_per_gpu: int = 16,
    global_batch_size_tokens: int = 500_000,
    warmup_tokens: int = 375_000_000,
    num_workers: int = 8,
    block_size: int = 2048,
    logging_level: int = logging.INFO,
    n_layer: int = 24,
    n_head: int = 24,
    n_embd: int = 2304,
    sparse_block_size: int = 128,
    stage: int = 3,
    local_rank: int = 0,
    precision: Union[str, int] = "bf16",
    log_dir: Optional[str] = None,
    save_every_n_steps: Optional[int] = None,
    save_dir: str = "checkpoints/",
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
        # todo: not totally correct, but we want to do a step per iteration for profiling.
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": precision == "bf16"},
        "fp16": {"enabled": precision == 16},
    }

    with deepspeed.zero.Init(config_dict_or_path=config):
        model = LLM(
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size,
            # LR scheduler only needs to worry about local device tokens processed
            warmup_tokens=int(warmup_tokens / num_devices),
            final_tokens=int((global_batch_size_tokens * num_iterations) / num_devices),
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            sparse_block_size=sparse_block_size,
        )

    optimizer, scheduler = model.configure_optimizers()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
        args=argparse.Namespace(device_rank=root_device.index),
        config=config,
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    train_loader = iter(train_loader)
    metrics = Metrics(
        log_dir=log_dir,
        engine=deepspeed_engine,
        iterations=num_iterations,
        batch_size=batch_size_per_gpu * num_devices,
        block_size=block_size,
        hidden_size=n_embd,
        n_layer=n_layer,
        vocab_size=train_dataset.vocab_size,
        num_devices=num_devices,
    )
    profiler = Profiler(
        global_batch_size=batch_size_per_gpu * num_devices,
        hidden_size=n_embd,
        num_devices=num_devices,
        device=root_device,
        n_layer=n_layer,
        block_size=block_size,
        vocab_size=train_dataset.vocab_size,
        activation_checkpointing=True
    )

    for x in range(num_iterations):
        if x == 10:
            profiler.start_profiling()
        batch = next(train_loader)
        src, targets = batch
        batch = (
            src.to(root_device, non_blocking=True),
            targets.to(root_device, non_blocking=True),
        )
        loss = deepspeed_engine(batch)
        if local_rank_zero:
            metrics.log(loss)
        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        if x == 20:
            profiler.end_profiling(10)



if __name__ == "__main__":
    fire.Fire(main)
