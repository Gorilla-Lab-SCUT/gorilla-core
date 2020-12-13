# Copyright (c) Gorilla-Lab. and its affiliates.
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import torch

from . import lr_scheduler as lr_schedulers
from ..core import is_seq_of
from ..config import Config

# the default optimizer and lr_scheduler config dict
OPTIM = {"name": "Adam",
         "lr": 0.001}

SCHEDULER = {"name": "StepLR",
             "step_size": 10000}


def build_single_optimizer(
        model: torch.nn.Module,
        optimizer_cfg: [Config, Dict]) -> torch.optim.Optimizer:
    r"""Author: zhang.haojian
    Build a single optimizer from optimizer config, supporting multi parameter
    groups with different setting in an optimizer
    """
    name = optimizer_cfg.pop("name")
    optimizer_cfg["params"] = []
    paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)
    if paramwise_cfg is None:
        # take the whole model parameters
        optimizer_cfg["params"].append(
            {"params": filter(lambda p: p.requires_grad, model.parameters())})
    else:
        for key, value in paramwise_cfg.items():
            optimizer_cfg["params"].append({
                "params":
                filter(lambda p: p.requires_grad,
                       getattr(model, key).parameters()),
                "name": key,
                **value
            })
    
    try:
        # a rich pytorch optimizer library
        # https://github.com/jettify/pytorch-optimizer
        import torch_optimizer
        optimizer_caller = getattr(torch_optimizer, name)
    except:
        optimizer_caller = getattr(torch.optim, name)
    
    return optimizer_caller(**optimizer_cfg)


def build_optimizer(model: torch.nn.Module,
                    optimizer_cfg: [Config, Dict]=OPTIM) -> torch.optim.Optimizer:
    r"""Author: zhang.haojian
    Build an optimizer from config, supporting multi optimizers.
    If there is no omission, build_optimizer_v2 can take the place of
    build_optimizer without changing the API
    Example:
        cfg = Config.fromfile(cfg.config_file)
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg.optimizer)
    """
    multi_optimizer = optimizer_cfg.pop("multi_optimizer", False)
    if multi_optimizer:
        optimizer_dict = {}
        for key, _optimizer_cfg in optimizer_cfg.items():
            optimizer_dict[key] = build_single_optimizer(model, _optimizer_cfg)
        return optimizer_dict
    else:
        _optimizer_cfg = optimizer_cfg
        return build_single_optimizer(model, _optimizer_cfg)


def build_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_scheduler_cfg: [Config, Dict]=SCHEDULER,
        lambda_func=None) -> torch.optim.lr_scheduler._LRScheduler:
    r"""Author: liang.zhihao
    Build a LR scheduler from config.

    Note:
        "name" must be in lr_scheduler_cfg

    Args:
        optimizer (torch.optim.Optimizer): Input Optimizer
        lr_scheduler_cfg ([Cofnig, Dict]): learning rate scheduler
        lambda_func(lambda, optional): Custom learning rate function,
                                       for using LambdaLR

    Example:
        cfg = Config.fromfile(cfg.config_file)
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_scheduler)

    Returns:
        _LRScheduler: the learning rate scheduler
    """
    name = lr_scheduler_cfg.pop("name")
    lr_scheduler_cfg["optimizer"] = optimizer
    if isinstance(optimizer, dict):
        # TODO: 暂不支持对多optimizer建立lr_scheduler，后面再想想怎么做
        return None

    # specificial for LambdaLR
    if name == "LambdaLR":
        assert lambda_func is not None
        assert isinstance(lambda_func, Callable) or \
            is_seq_of(lambda_func, Callable), "lambda_func is invalid"
        lr_scheduler_cfg["lr_lambda"] = lambda_func

    # get the caller
    scheduler_caller = getattr(lr_schedulers, name)
    return scheduler_caller(**lr_scheduler_cfg)

