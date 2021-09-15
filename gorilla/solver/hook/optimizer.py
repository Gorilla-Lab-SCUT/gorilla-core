# Copyright (c) Open-MMLab. All rights reserved.
import copy

from torch.nn.utils import clip_grad

from .hook import Hook
from ...core import HOOKS


@HOOKS.register()
class OptimizerHook(Hook):
    # TODO: add comment
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip
        # self._content = dict(after_step="loss backward and optimizer step")

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_step(self, solver):
        r"""loss backward and optimizer step"""
        solver.optimizer.zero_grad()
        loss = solver.outputs["loss"]
        loss.backward()
        solver.optimizer.step()
        solver.lr_scheduler.step()
