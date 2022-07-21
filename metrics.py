import time
from decimal import Decimal
from typing import Dict

import torch
from deepspeed import DeepSpeedEngine
from torch.utils.tensorboard import SummaryWriter


class Metrics:
    def __init__(
        self,
        log_dir: str,
        engine: DeepSpeedEngine,
        iterations: int,
        batch_size: int,
        block_size: int,
        hidden_size: int,
        num_devices: int,
        n_layer: int,
        vocab_size: int,
    ):
        self.iteration = -1
        self.per_iteration_time = 0
        self.start = time.time()
        self.logger = SummaryWriter(log_dir=log_dir, flush_secs=1)
        self.engine = engine
        self.batch_size = batch_size
        self.block_size = block_size
        self.iterations = iterations
        self.tflops_estimator = TFLOPs(
            global_batch_size=batch_size,
            hidden_size=hidden_size,
            num_devices=num_devices,
            n_layer=n_layer,
            block_size=block_size,
            vocab_size=vocab_size,
        )

    def log(self, loss: torch.tensor) -> None:
        self.per_iteration_time = time.time() - self.start
        self.iteration += 1
        logger_dict = self.logger_dict(loss)
        # log to tensorboard
        for key, value in logger_dict.items():
            self.logger.add_scalar(key, value, global_step=self.iteration)
        # print to stdout
        logger_dict["learning_rate"] = "%.3E" % Decimal(self.learning_rate)
        logger_string = " | ".join(f"{k} : {v}" for k, v in logger_dict.items())
        logger_string = f"[{self.iteration}/{self.iterations}] | secs/it {self.per_iteration_time:.2f} | {logger_string}"
        print(logger_string)
        self.start = time.time()

    def logger_dict(self, loss: torch.tensor) -> Dict:
        return {
            "loss": float(loss.item()),
            "TFLOPs": self.tflops,
            "learning_rate": self.learning_rate,
            "consumed_samples": self.consumed_samples,
            "consumed_tokens": self.consumed_tokens,
            # todo: we should add grad norm
        }

    @property
    def tflops(self) -> float:
        return int(self.tflops_estimator.calculate_flops(self.per_iteration_time))

    @property
    def learning_rate(self) -> float:
        return self.engine.get_lr()[0]

    @property
    def consumed_samples(self) -> int:
        return self.batch_size * (self.iteration + 1)

    @property
    def consumed_tokens(self) -> int:
        return self.consumed_samples * self.block_size


class TFLOPs:
    def __init__(
        self,
        global_batch_size: int,
        hidden_size: int,
        num_devices: int,
        n_layer: int,
        block_size: int,
        vocab_size: int,
        activation_checkpointing: bool = True,
    ):
        self.num_devices = num_devices
        # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
        # https://arxiv.org/pdf/2104.04473.pdf).
        # https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251/files
        self.num_parameters = (
            n_layer * (12 * hidden_size ** 2 + 13 * hidden_size)
            + vocab_size * hidden_size
            + block_size * hidden_size
            + 2 * hidden_size
        ) / 10 ** 9
        factor = 4 if activation_checkpointing else 3
        self.flops_per_iteration = (
            24 * factor * global_batch_size * block_size * n_layer * (hidden_size ** 2)
        ) * (
            1.0
            + (block_size / (6.0 * hidden_size))
            + (vocab_size / (16.0 * n_layer * hidden_size))
        )

    def calculate_flops(self, per_iteration_time: float) -> float:
        return self.flops_per_iteration / (
            per_iteration_time * self.num_devices * (10 ** 12)
        )
