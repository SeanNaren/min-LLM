import psutil
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class CUDAMemoryCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))

    def root_gpu(self, trainer):
        return trainer.strategy.root_device.index

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2 ** 20
        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        swap = psutil.swap_memory()
        swap = round((swap.used / (1024 ** 3)), 2)

        max_memory = trainer.strategy.reduce(max_memory)
        virt_mem = trainer.strategy.reduce(virt_mem)
        swap = trainer.strategy.reduce(swap)

        rank_zero_info(f"Average Peak CUDA memory {max_memory:.2f} MiB")
        rank_zero_info(f"Average Peak Virtual memory {virt_mem:.2f} GiB")
        rank_zero_info(f"Average Peak Swap memory {swap:.2f} Gib")
