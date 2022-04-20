from typing import Any

import pytorch_lightning as pl
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler, MODULE_HOOK_MAPPING
from pytorch_lightning import Callback
from xformers.components.attention import BlockSparseAttention
from xformers.triton import FusedLayerNorm, FusedLinear


def _layer_norm_forward_hook(layer_norm_module: FusedLayerNorm, input, output):
    input = input[0]
    has_affine = layer_norm_module.weight is not None
    flops = torch.numel(input) * (5 if has_affine else 4)
    layer_norm_module.__flops__ += int(flops)


def _linear_forward_hook(linear_module: FusedLinear, input, output):
    input = input[0]
    out_features = linear_module.weight.shape[0]
    macs = torch.numel(input) * out_features
    flops = 2 * macs
    linear_module.__flops__ += int(flops)


def _block_sparse_forward_hook(sparse_attention: BlockSparseAttention, input, output):
    q, k, v = input
    sparse_att_mat_flops = torch.prod(torch.tensor(q.shape)) * k.shape[-1]
    q_k_shape = torch.tensor([q.shape[0], q.shape[1], q.shape[2], q.shape[2]])
    softmax_flops = torch.prod(q_k_shape)
    a_flops = torch.prod(q_k_shape) * v.shape[-1]
    flops = sparse_att_mat_flops + softmax_flops + a_flops
    sparse_attention.__flops__ += int(flops)


# these are additional hooks that the profiler needs to properly profile these modules.
# since they are custom (not native pytorch modules) they need to be defined.
ADDITIONAL_MODULE_HOOK_MAPPING = {
    FusedLayerNorm: _layer_norm_forward_hook,
    FusedLinear: _linear_forward_hook,
    BlockSparseAttention: _block_sparse_forward_hook
}


class DeepSpeedProfiler(Callback):

    def __init__(self, start_idx: int = 20, end_idx: int = 40):
        MODULE_HOOK_MAPPING.update(ADDITIONAL_MODULE_HOOK_MAPPING)
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.prof = None

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if trainer.is_global_zero:
            if self.prof is None:
                self.prof = FlopsProfiler(pl_module)
            if batch_idx == self.start_idx:
                self.prof.start_profile()

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if trainer.is_global_zero and (batch_idx == self.end_idx):
            self.prof.print_model_profile(batch_idx, detailed=False)
