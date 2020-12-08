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
                 logger=None):
        # initial essential parameters
        self.model = model
        self.optimizer = build_optimizer(model, cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg.lr_scheduler)
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.logger = logger
        self.epoch = cfg.get("start_epoch", 1)
        self.log_buffer = LogBuffer()
        self.tb_writer = SummaryWriter(log_dir=cfg.log) # tensorboard writer
        self.iter = 0  # cumulative iter number, doesn't flush when come into a new epoch

        # the hooks container (optional)
        self._hooks = []

        self.get_ready()

    def get_ready(self):
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
        self.clear()
        # the whole training processing

    @abstractmethod
    def train(self):
        r"""train(self) aims to define each step training operation"""
        self.clear()
        # epoch training

    @abstractmethod
    def evaluate(self):
        r"""evaluate(self) aims to define each evaluation operation"""
        self.clear()
        # evaluation

    def clear(self):
        r"""clear log buffer
        """
        self.log_buffer.clear()

    def write(self):
        self.logger.info("Epoch: {}".format(self.epoch))
        self.log_buffer.average()
        for key, avg in self.log_buffer.output.items():
            self.tb_writer.add_scalar(key, avg, self.epoch)

    def resume(self, checkpoint):
        check_file_exist(checkpoint)
        resume(self.model,
               checkpoint,
               self.optimizer,
               self.lr_scheduler)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_max_memory(self):
        device = list(self.model.parameters())[0].device
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        return mem_mb.item()

    def quit(self):
        self.tb_writer.close()
