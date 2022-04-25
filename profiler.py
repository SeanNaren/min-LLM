import time
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info


class GPTFLOPsEstimate(Callback):
    """
    This callback wraps the function described in the Megatron-lm paper and the BigScience README
    to calculate the lower bound estimated FLOPs for a GPT model:

    https://arxiv.org/abs/2104.04473
    https://github.com/bigscience-workshop/bigscience/tree/master/math#calculate-tflops
    https://github.com/bigscience-workshop/bigscience/blob/master/experiments/gpt2-utils.md#calculate-model-size
    """

    def __init__(self,
                 global_batch_size: int,
                 hidden_size: int,
                 n_layer: int,
                 block_size: int,
                 vocab_size: int,
                 activation_checkpointing: bool = False,
                 profile_start_step: int = 20,
                 profile_num_steps: int = 40):
        self.profile_start_step = profile_start_step
        self.profile_num_steps = profile_num_steps
        self.global_batch_size = global_batch_size
        self.activation_checkpointing = activation_checkpointing
        h = hidden_size
        l = n_layer
        self.s = block_size
        v = vocab_size

        self.num_parameters = (l * (12 * h ** 2 + 13 * h) + v * h + self.s * h + 2 * h) / 10 ** 9
        print(f"Number of parameters: {self.num_parameters:.2f} Billion")

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if trainer.is_global_zero and batch_idx == self.profile_start_step:
            self.start = time.time()

    @property
    def num_steps(self):
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
        if trainer.is_global_zero and (batch_idx == self.num_steps):
            total_time = time.time() - self.start
            factor = 4 if self.activation_checkpointing else 3
            num_steps = self.profile_num_steps - self.profile_start_step
            per_iteration_time = total_time / num_steps
            gpus = trainer.devices
            flops = self.num_parameters * factor * 2 * self.s * self.global_batch_size
            flops = flops / (per_iteration_time * gpus * 1e3)
            rank_zero_info(f"Estimates: {flops:.2f}TFLOPs Avg Iteration Time: {per_iteration_time:.2f}s")
