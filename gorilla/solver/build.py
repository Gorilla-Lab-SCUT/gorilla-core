# Copyright (c) Gorilla-Lab. and its affiliates.
from typing import Callable, Dict

import torch

from .data import DataLoaderX
from ..config import Config
from ..core import is_seq_of, _build_optimizer, _build_scheduler, build_dataset

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
    # name = optimizer_cfg.pop("name")
    # get params
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
                filter(lambda p: p.requires_grad, getattr(model, key).parameters()),
                "name": key,
                **value
            })
    
    return _build_optimizer(optimizer_cfg)


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
    lr_scheduler_cfg["optimizer"] = optimizer
    if isinstance(optimizer, dict):
        # TODO: do not support build multi lr_schedulers for multi optimizer
        raise NotImplementedError

    # specificial for LambdaLR
    if lr_scheduler_cfg.get("name") == "LambdaLR":
        assert lambda_func is not None
        assert isinstance(lambda_func, Callable) or \
            is_seq_of(lambda_func, Callable), "lambda_func is invalid"
        lr_scheduler_cfg["lr_lambda"] = lambda_func

    return _build_scheduler(lr_scheduler_cfg)


# TODO: support distributed and multi dataloader
def build_dataloader(
    dataset: [torch.utils.data.Dataset, Dict],
    dataloader_cfg: Dict,
    prefetch: bool=False,
    **kwargs) -> torch.utils.data.DataLoader:
    """Author: liang.zhihao
    Support callback "collate_fn" defined in dataset

    Args:
        dataset ([torch.utils.data.Dataset, Dict]): input dataset object of config dict
        dataloader_cfg (Dict): config dict for building dataloader
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        torch.utils.data.DataLoader: output dataloader
    """
    if isinstance(dataset, Dict):
        dataset = build_dataset(dataset)

    dataloader_cfg.update(kwargs)
    distribute = distributed_prepare(dataloader_cfg)

    collate_fn = getattr(dataset, "collate_fn", None)
    dataloader_cfg["collate_fn"] = collate_fn
    assert "batch_size" in dataloader_cfg, "must given batch_size"
    assert "num_workers" in dataloader_cfg, "must given num_workers"

    if prefetch:
        return DataLoaderX(dataset, **dataloader_cfg)
    else:
        return torch.utils.data.DataLoader(dataset, **dataloader_cfg)


def distributed_prepare(dataloader_cfg):
    # TODO: implement
    return None
