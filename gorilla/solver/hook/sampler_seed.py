# Copyright (c) Open-MMLab. All rights reserved.
from .hook import Hook
from ...core import HOOKS


@HOOKS.register_module()
class DistSamplerSeedHook(Hook):

    def before_epoch(self, solver):
        # if isinstance(solver.dataloader.sampler, list):
        #     for s in solver.dataloader.sampler:
        #         s.set_epoch(solver.epoch)
        # else:
        solver.dataloader.sampler.set_epoch(solver.epoch)
