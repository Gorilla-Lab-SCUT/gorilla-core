# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import random
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from tensorboardX import SummaryWriter

from .log_buffer import LogBuffer
from .build import build_optimizer, build_lr_scheduler
from .checkpoint import resume
from ..utils import check_file_exist


class BaseSolver(metaclass=ABCMeta):
    r"""Base class of model solver."""
    def __init__(self,
                 model,
                 dataloaders,
                 cfg,
                 logger=None,
                 **kwargs):
        # initial essential parameters
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = build_optimizer(model, cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg.lr_scheduler)
        self.cfg = cfg
        self.logger = logger
        self.epoch = cfg.get("start_epoch", 1)
        self.log_buffer = LogBuffer()
        self.tb_writer = SummaryWriter(log_dir=cfg.log_dir) # tensorboard writer
        self.iter = 0  # cumulative iter number, doesn't flush when come into a new epoch
        self.meta = {}

        self.get_ready()

    def get_ready(self, **kwargs):
        pass

    def resume(self, checkpoint, **kwargs):
        check_file_exist(checkpoint)
        self.meta = resume(self.model,
                           checkpoint,
                           self.optimizer,
                           self.lr_scheduler,
                           **kwargs)
        self.logger.info("resume from: {}".format(checkpoint))
        if "epoch" in self.meta:
            self.epoch = self.meta["epoch"] + 1

    def write(self, **kwargs):
        self.logger.info("Epoch: {}".format(self.epoch))
        self.log_buffer.average()
        for key, avg in self.log_buffer.output.items():
            self.tb_writer.add_scalar(key, avg, self.epoch)

    def clear(self, **kwargs):
        r"""clear log buffer
        """
        self.log_buffer.clear()

    @abstractmethod
    def solve(self, **kwargs):
        r"""solve(self) aims to define each epoch training operation"""
        self.clear()
        # the whole training processing

    @abstractmethod
    def train(self, **kwargs):
        r"""train(self) aims to define each step training operation"""
        self.clear()
        # epoch training

    @abstractmethod
    def evaluate(self, **kwargs):
        r"""evaluate(self) aims to define each evaluation operation"""
        self.clear()
        # evaluation


