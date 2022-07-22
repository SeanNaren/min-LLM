import time

import torch


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
