import argparse
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(
    num_iterations: int = 50000,
    batch_size_per_gpu: int = 16,
    num_workers: int = 8,
    block_size: int = 2048,
    warmup: int = 20,
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

    train_loader = iter(train_loader)
    metrics = Metrics(
        log_dir=log_dir,
        deepspeed_engine=deepspeed_engine,
        iterations=num_iterations,
        batch_size=global_batch_size,
        block_size=block_size,
        hidden_size=n_embd,
        n_layer=n_layer,
        vocab_size=train_dataset.vocab_size,
        num_devices=num_devices,
    )

    for x in range(num_iterations):
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

        if save_every_n_steps and x > 0 and x % save_every_n_steps == 0:
            deepspeed_engine.save_checkpoint(save_dir)


if __name__ == "__main__":
    fire.Fire(main)
