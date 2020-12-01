# Copyright (c) Facebook, Inc. and its affiliates.
from yacs.config import CfgNode
import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineLR, WarmupPolyLR

def bulid_solver(model, dataloaders, optimizer, lr_scheduler, cfg):
    if cfg.method == "DANN":
        from .solvers.solver_dann import solver_DANN
        return solver_DANN(model, dataloaders, optimizer, lr_scheduler, cfg)


def build_optimizer(cfg: CfgNode, model: torch.nn.Module, optimizer_type=None) -> torch.optim.Optimizer:
    r"""
    Build an optimizer from config.
    """
    if optimizer_type is None:
        optimizer_type = cfg.OPTIMIZER_TYPE

    if optimizer_type == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            cfg.BASE_LR,
            momentum=cfg.MOMENTUM,
            nesterov=cfg.NESTEROV
        )
    elif optimizer_type == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            cfg.BASE_LR,
            weight_decay=cfg.WEIGHT_DECAY,
            amsgrad=cfg.AMSGRAD)
    elif optimizer_type == "AdamW":
        return torch.optim.Adam(
            model.parameters(),
            cfg.BASE_LR,
            betas=cfg.BETAS,
            weight_decay=cfg.WEIGHT_DECAY,
            amsgrad=cfg.AMSGRAD)
    else:
        raise NotImplementedError("no optimizer type {}".format(optimizer_type))


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer, lr_scheduler_name: str=None
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Build a LR scheduler from config.
    """
    if lr_scheduler_name is None:
        name = cfg.LR_SCHEDULER_NAME
    else:
        name = lr_scheduler_name
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.STEPS,
            cfg.GAMMA,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.MAX_ITER,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
        )
    elif name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.MAX_ITER,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
            power=cfg.POLY_LR_POWER,
            constant_ending=cfg.POLY_LR_CONSTANT_ENDING,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
