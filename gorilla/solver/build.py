# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

from ..config import Config
import torch

from . import lr_scheduler
from .base_solver import BaseSolver

from ..core import is_seq_of


def bulid_solver(model, dataloaders, optimizer, lr_scheduler, cfg):
    return BaseSolver(model, dataloaders, optimizer, lr_scheduler, cfg)


def build_optimizer(cfg: [Config, Dict], model: torch.nn.Module, optimizer_type=None) -> torch.optim.Optimizer:
    r"""
    Build an optimizer from config.
    """
    if optimizer_type is not None:
        cfg["optimizer_type"] = optimizer_type

    optimizer_type = cfg.pop("optimizer_type")

    cfg["params"] = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_caller = getattr(torch.optim, optimizer_type)
    return optimizer_caller(**cfg)


def build_lr_scheduler(
    cfg: [Config, Dict], optimizer: torch.optim.Optimizer, lr_scheduler_name: str=None, lambda_func=None
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Build a LR scheduler from config.
    """
    if lr_scheduler_name is not None:
        cfg["lr_scheduler_name"] = lr_scheduler_name
    
    lr_scheduler_name = cfg.pop("lr_scheduler_name")
    cfg["optimizer"] = optimizer

    # specificial for LambdaLR
    if lr_scheduler_name == "LambdaLR":
        assert lambda_func is not None
        assert isinstance(lambda_func, Callable) or \
            is_seq_of(lambda_func, Callable), "lambda_func is invalid"
        cfg["lr_lambda"] = lambda_func
    
    scheduler_caller = getattr(lr_scheduler, lr_scheduler_name)
    return scheduler_caller(**cfg)

