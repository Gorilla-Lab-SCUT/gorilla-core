# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

from ..config import Config
import torch

from . import lr_scheduler as lr_schedulers
from .base_solver import BaseSolver

from ..core import is_seq_of


def bulid_solver(model, optimizer, dataloaders, lr_scheduler, cfg, logger=None):
    return BaseSolver(model,
                      optimizer,
                      dataloaders,
                      lr_scheduler,
                      cfg,
                      logger)


def build_optimizer(model: torch.nn.Module,
                    cfg: [Config, Dict]) -> torch.optim.Optimizer:
    r"""
    Build an optimizer from config.
    """
    optimizer_type = cfg.pop("optimizer_type")

    cfg["params"] = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_caller = getattr(torch.optim, optimizer_type)
    return optimizer_caller(**cfg)


def build_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: [Config, Dict],
        lambda_func=None) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Build a LR scheduler from config.
    """

    lr_scheduler_name = cfg.pop("lr_scheduler_name")
    cfg["optimizer"] = optimizer

    # specificial for LambdaLR
    if lr_scheduler_name == "LambdaLR":
        assert lambda_func is not None
        assert isinstance(lambda_func, Callable) or \
            is_seq_of(lambda_func, Callable), "lambda_func is invalid"
        cfg["lr_lambda"] = lambda_func

    scheduler_caller = getattr(lr_schedulers, lr_scheduler_name)
    return scheduler_caller(**cfg)
