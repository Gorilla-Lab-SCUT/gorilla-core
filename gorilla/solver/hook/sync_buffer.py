# Copyright (c) Open-MMLab. All rights reserved.
import torch.distributed as dist

from .hook import Hook
from ...core import HOOKS, get_dist_info


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    r"""Synchronize model buffers such as running_mean and running_var in BN at
        the end of each epoch.

    Args:
        distributed (bool): Whether distributed training is used. It is
          effective only for distributed training. Defaults to True.
    """

    def __init__(self, distributed=True):
        self.distributed = distributed

    def after_epoch(self, solver):
        r"""All-reduce model buffers at the end of each epoch."""
        _, world_size = get_dist_info()
        if self.distributed and world_size > 1:
            buffers = solver.model.buffers()
            world_size = dist.get_world_size()
            for tensor in buffers:
                dist.all_reduce(tensor.div_(world_size))
