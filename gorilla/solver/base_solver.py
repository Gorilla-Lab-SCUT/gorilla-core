# Copyright (c) Gorilla-Lab. All rights reserved.
import logging
import os
import random
from abc import ABCMeta, abstractmethod

import torch

from .log_buffer import LogBuffer, TensorBoardWriter
from .build import build_optimizer, build_lr_scheduler
from .checkpoint import resume
from ..utils import check_file
from ..core import build_model


class BaseSolver(metaclass=ABCMeta):
    r"""Base class of model solver."""
    def __init__(self, model, dataloaders, cfg, **kwargs):
        # TODO: the model builder is ugly and need to
        # integrate into solver elegant
        if isinstance(model, dict):
            self.model = build_model(model)
        elif isinstance(model, torch.nn.Module):
            self.model = model
        else:
            raise TypeError(
                f"`model` must be `nn.module` or cfg `dict`, but got `{type(model)}`"
            )

        # get the epoch or iteration mode
        self.mode = cfg.get("mode", "epoch")
        assert self.mode in ["epoch", "iter"]

        # merge some essentital parameter for each project(like independent criterion)
        self.__dict__.update(kwargs)

        # initial essential parameters
        self.dataloaders = dataloaders
        self.optimizer = build_optimizer(model, cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer,
                                               cfg.lr_scheduler)
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        self.get_ready()
        self.callback()

    def get_ready(self, **kwargs):
        self.epoch = self.cfg.get("start_epoch", 1)
        self.tb_writer = TensorBoardWriter(
            self.cfg.log_dir)  # tensorboard writer
        self.log_buffer = self.tb_writer.buffer  # to fix old API
        self.iter = 0  # cumulative iter number, doesn't flush when come into a new epoch
        self.meta = {}

    def callback(self, **kwargs):
        """a callback function for self defined
        """
        pass

    def resume(self, checkpoint, **kwargs):
        check_file(checkpoint)
        self.meta = resume(self.model, checkpoint, self.optimizer,
                           self.lr_scheduler, **kwargs)
        if "epoch" in self.meta:
            self.epoch = self.meta["epoch"] + 1
        if "iter" in self.meta:
            self.iter = self.meta["iter"] + 1

    def write(self, **kwargs):
        if self.mode == "epoch":
            times = self.epoch
        elif self.mode == "iter":
            times = self.iter
        self.tb_writer.write(self.epoch)
        # self.log_buffer.average()
        # for key, avg in self.log_buffer.output.items():
        #     self.tb_writer.add_scalar(key, avg, times)

    def clear(self, **kwargs):
        r"""clear log buffer
        """
        self.log_buffer.clear()

    def solve(self, **kwargs):
        r"""solve(self) aims to define each epoch training operation"""
        self.clear()
        # the whole training processing

    def train(self, **kwargs):
        r"""train(self) aims to define each step training operation"""
        self.clear()
        # epoch training

    def evaluate(self, **kwargs):
        r"""evaluate(self) aims to define each evaluation operation"""
        self.clear()
        # evaluation

    # TODO: support the hook
    def register_hook(self):
        from .hook import HookManager
        self.hook_manager = HookManager()
        self.hook_manager.concat_solver(self)
        self.hook_manager.register_hook_from_cfg(dict(type="OptimizerHook"))
        self.hook_manager.register_hook_from_cfg(dict(type="EmptyCacheHook"))
        self.hook_manager.register_hook_from_cfg(dict(type="IterTimerHook"))
        self.hook_manager.register_hook_from_cfg(dict(type="CheckpointHook"))
        # self.logger.info(self.hook_manager)

    # def call_hook(self, fn_name):
    #     r"""Call all hooks.

    #     Args:
    #         fn_name (str): The function name in each hook to be called, such as
    #             "before_epoch".
    #     """
    #     for hook in self._hooks:
    #         getattr(hook, fn_name)(self)
