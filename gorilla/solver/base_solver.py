# Copyright (c) Gorilla-Lab. All rights reserved.
from abc import ABCMeta, abstractmethod
import os
import torch
import random
import numpy as np

# import SummaryWriter
if torch.__version__ < "1.1":
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError("Please install tensorboardX "
                            "to use Tensorboard.")
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError("Please run 'pip install future tensorboard' to install "
                          "the dependencies to use torch.utils.tensorboard "
                          "(applicable to PyTorch 1.1 or higher)")

from .log_buffer import LogBuffer

class BaseSolver(metaclass=ABCMeta):
    r"""Base class of model solver."""
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, cfg, logger=None):
        # initial essential parameters
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_dataloader
        self.val_data_loader = val_dataloader
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
        else: # do not set random seed
            pass
    
    @property
    def complete_training_flag(self):
        return self.iter > self.cfg.max_iters or \
            self.epoch > self.cfg.max_epoch

    @abstractmethod
    def solve(self):
        r"""solve(self) aims to define each epoch training operation"""
        pass

    @abstractmethod
    def train(self):
        r"""train(self) aims to define each step training operation"""
        pass

    @abstractmethod
    def test(self):
        r"""test aims to define each evaluation operation"""
        pass

    def quit(self):
        self.writer.close()
