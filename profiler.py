import time
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class GPTFLOPsEstimate(Callback):
    """
    This callback wraps the function described in the Megatron-lm paper and the BigScience README
    to calculate the lower bound estimated FLOPs for a GPT model:

    https://arxiv.org/abs/2104.04473
    https://github.com/bigscience-workshop/bigscience/tree/master/math#calculate-tflops
    https://github.com/bigscience-workshop/bigscience/blob/master/experiments/gpt2-utils.md#calculate-model-size
    """

    def __init__(
        self,
        global_batch_size: int,
        hidden_size: int,
        n_layer: int,
        block_size: int,
        vocab_size: int,
        activation_checkpointing: bool = False,
        profile_start_step: int = 20,
        profile_num_steps: int = 20,
    ):
        self.profile_start_step = profile_start_step
        self.profile_num_steps = profile_num_steps
        self.global_batch_size = global_batch_size
        self.activation_checkpointing = activation_checkpointing
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
        rank_zero_info(f"Number of parameters: {self.num_parameters:.2f} Billion")

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if trainer.is_global_zero and batch_idx == self.profile_start_step:
            torch.cuda.synchronize()
            self.start = time.time()

    @property
    def profile_end_step(self):
        return self.profile_num_steps + self.profile_start_step

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if trainer.is_global_zero and (batch_idx == self.profile_end_step):
            torch.cuda.synchronize()
            total_time = time.time() - self.start
            factor = 4 if self.activation_checkpointing else 3
            per_iteration_time = total_time / self.profile_num_steps
            # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
            # https://arxiv.org/pdf/2104.04473.pdf).
            # https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251/files
            flops_per_iteration = (
                24 * factor * self.global_batch_size * self.s * self.l * (self.h ** 2)
            ) * (1.0 + (self.s / (6.0 * self.h)) + (self.v / (16.0 * self.l * self.h)))
            flops = flops_per_iteration / (
                per_iteration_time * trainer.num_devices * (10 ** 12)
            )
            rank_zero_info(
                f"Estimates: {flops:.2f}TFLOPs Avg Iteration Time: {per_iteration_time:.2f}s"
            )
