# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import torch
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from tensorboardX import SummaryWriter

from .log_buffer import LogBuffer


class BaseSolver(metaclass=ABCMeta):
    r"""Base class of model solver."""
    def __init__(self,
                 model,
                 optimizer,
                 dataloaders,
                 lr_scheduler,
                 cfg,
                 logger=None):
        # initial essential parameters
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.epoch = cfg.start_epoch
        self.logger = logger
        self.writer = SummaryWriter(log_dir=cfg.log)
        self.iter = 0  # cumulative iter number, doesn't flush when come into a new epoch
        self.log_buffer = LogBuffer()

        # the hooks container (optional)
        self._hooks = []

        self.do_before_training()

    def do_before_training(self):
        # set random seed to keep the result reproducible
        if self.cfg.seed != 0:
            from ..core import set_random_seed
            print("set random seed:", self.cfg.seed)
            set_random_seed(self.cfg.seed)
        else:  # do not set random seed
            pass

    @property
    def complete_training_flag(self):
        return self.iter > self.cfg.max_iters or \
            self.epoch > self.cfg.max_epoch

    @abstractmethod
    def solve(self):
        r"""solve(self) aims to define each epoch training operation"""
        pass
        # the whole training processing

    @abstractmethod
    def train(self):
        r"""train(self) aims to define each step training operation"""
        self.log_buffer.clear()
        # epoch training

    @abstractmethod
    def evaluate(self):
        r"""evaluate(self) aims to define each evaluation operation"""
        self.log_buffer.clear()
        # evaluation

    def write(self):
        self.log_buffer.average()
        for key, avg in self.log_buffer.output.items():
            self.writer.add_scalar(key, avg, self.epoch)

    def quit(self):
        self.writer.close()
