from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
import os
import torch
import random
import numpy as np

# from utils.logger import Logger

class BaseSolver(ABC):
    r"""Base class of model solver."""
    def __init__(self, models, optimizers, dataloaders, lr_scheduler, cfg):
        self.models = models
        self.optimizers = optimizers
        self.data_loaders = dataloaders
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self._hooks = []
        self.epoch = cfg.start_epoch
        self.iter = 0 # cumulative iter number, doesn't flush when come into a new epoch
        self.best_prec1 = 0
        # self.iters_per_epoch = None # it will be defined in subclass of BaseSolver
        if cfg.iters_per_epoch == 0:
            self.iters_per_epoch = len(self.data_loaders["train_src"])
        else:
            self.iters_per_epoch = cfg.iters_per_epoch
        self.do_before_training()
        self.writer = SummaryWriter(log_dir=cfg.log)

    def do_before_training(self):
        # init data iterators
        data_iterators = {}
        for key in self.data_loaders.keys():
            data_iterators[key] = iter(self.data_loaders[key])
        self.data_iterators = data_iterators

        for_reproduce = True # set False may speed up a little
        if for_reproduce:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # set random seed to keep the result reproducible
        if self.cfg.seed != 0:
            from gorilla.utils import set_seed
            print("set random seed:", self.cfg.seed)
            set_seed(self.cfg.seed)
        else: # do not set random seed
            pass

    def get_samples(self, data_name):
        assert(data_name in self.data_loaders)

        data_loader = self.data_loaders[data_name]
        data_iterator = self.data_iterators[data_name]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.data_iterators[data_name] = data_iterator
        return sample

    def complete_training(self):
        if self.iter > self.cfg.max_iters:
            return True

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def quit(self):
        self.writer.close()
