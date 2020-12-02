# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

from yacs.config import CfgNode
import torch

from .lr_scheduler import (CosineAnnealingLR, CyclicLR, ExponentialLR,
                           MultiStepLR, OneCycleLR, StepLR, LambdaLR, PolyLR,
                           WarmupMultiStepLR, WarmupCosineLR, WarmupPolyLR)
                           
from ..core import is_seq_of


def bulid_solver(model, dataloaders, optimizer, lr_scheduler, cfg):
    if cfg.method == "DANN":
        from .solvers.solver_dann import solver_DANN
        return solver_DANN(model, dataloaders, optimizer, lr_scheduler, cfg)


def build_optimizer(cfg: [CfgNode, Dict], model: torch.nn.Module, optimizer_type=None) -> torch.optim.Optimizer:
    r"""
    Build an optimizer from config.
    """
    if optimizer_type is not None:
        if isinstance(cfg, CfgNode):
            cfg.optimizer_type = optimizer_type
            cfg = dict(cfg)
        elif isinstance(cfg, Dict):
            cfg["optimizer_type"] = optimizer_type
        else:
            raise TypeError("cfg must be CfgNode or Dict, but got {}".format(type(cfg)))

    optimizer_type = cfg.pop("optimizer_type")

    cfg["params"] = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_caller = getattr(torch.optim, optimizer_type)
    return optimizer_caller(**cfg)


def build_lr_scheduler(
    cfg: [CfgNode, Dict], optimizer: torch.optim.Optimizer, lr_scheduler_name: str=None, lambda_func=None
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Build a LR scheduler from config.
    """
    if lr_scheduler_name is not None:
        if isinstance(cfg, CfgNode):
            cfg.lr_scheduler_name = lr_scheduler_name
            cfg = dict(cfg)
        elif isinstance(cfg, Dict):
            cfg["lr_scheduler_name"] = lr_scheduler_name
        else:
            raise TypeError("cfg must be CfgNode or Dict, but got {}".format(type(cfg)))
    
    lr_scheduler_name = cfg.pop("lr_scheduler_name")
    cfg["optimizer"] = optimizer

    # specificial for LambdaLR
    if lr_scheduler_name == "LambdaLR":
        assert lambda_func is not None
        assert isinstance(lambda_func, Callable) or \
            is_seq_of(lambda_func, Callable), "lambda_func is invalid"
        cfg["lr_lambda"] = lambda_func
    
    scheduler_caller = globals()[lr_scheduler_name]
    return scheduler_caller(**cfg)

