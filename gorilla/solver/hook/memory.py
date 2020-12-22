# Copyright (c) Open-MMLab. All rights reserved.
import torch

from gorilla.core import HOOKS
from .hook import Hook


@HOOKS.register_module()
class EmptyCacheHook(Hook):

    def __init__(self,
                 after_step=True,
                 after_epoch=True):
        self._after_step = after_step
        self._after_epoch = after_epoch

    def after_step(self, solver):
        r"""clean cuda memory"""
        if self._after_step:
            torch.cuda.empty_cache()

    def after_epoch(self, solver):
        r"""clean cuda memoty"""
        if self._after_epoch:
            torch.cuda.empty_cache()

