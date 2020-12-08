# Copyright (c) Facebook, Inc. and its affiliates.
import math
from bisect import bisect_right
from typing import List
import torch

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.

#### pytorch base lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, CyclicLR, ExponentialLR,
                                      MultiStepLR, OneCycleLR, LambdaLR)


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.constant_ending > 0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class StepLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] * group.get("lr_multi", 1.0)
                for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma * group.get("lr_multi", 1.0)
            for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class InvLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Note p as current epoch or iter number according to user setting;
    Note maxp as num_epochs or num_iters according to user setting;
    p / maxp ranges from [0, 1].
    Then lr(p) = base_lr * (1 + gamma * p / maxp)**(-power).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        maxp (int): Maximum epochs (or maximum iterations if
            lr_scheduler.step() is called each iteration)
        gamma (float): Default: 0.1.
        power (float): Default: 0.75.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> scheduler = InvLR(optimizer, gamma=10, power=0.75)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, maxp, gamma=10, power=0.75, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        self.maxp = maxp
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        choice = 2
        if choice == 1:
            # implementation 1 (mine): fixed gamma(10), the shape of lr-iter curve will keep
            # the same although maxp are different
            progress = self.last_epoch / self.maxp
            return [
                base_lr * (1 + self.gamma * progress)**(-self.power) *
                group.get("lr_mult", 1.0) for group, base_lr in zip(
                    self.optimizer.param_groups, self.base_lrs)
            ]
        elif choice == 2:
            # implementation 2 (Long Mingsheng): fixed gamma(0.001), the same last_epoch will
            # lead to the same lr, although maxp are different
            return [
                base_lr * (1 + self.gamma * self.last_epoch)**(-self.power) *
                group.get("lr_mult", 1.0) for group, base_lr in zip(
                    self.optimizer.param_groups, self.base_lrs)
            ]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 or warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters or warmup_iters == 0:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def adjust_learning_rate(optimizer, epoch, args, mode="auto", value=0.1, namelist=[]):
    r"""
    Adjust the learning rate according to the epoch
    Parameters
    ----------
    optimzer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model
    epoch: int
        The current epoch
    args: Namespace
        Arguments that main.py receive
    total_epochs: int
        The total epoch number
    mode: str
        Mode of setting lr, 'auto' (computed automatically by formula), 'rel' (relative) or 'abs' (absolute)
    value: float
        In 'auto' mode, lr of pretrained modules additionally multiply this variable;
        In 'rel' mode, parameters multiply this variable;
        In 'abs' mode, parameters is set to this variable
    namelist: list
        If namelist is not empty, then only adjust the lr of param_groups whose name is in namelist;
        If namelist is empty (default), then adjust the lr of all param_group
    Return
    ------
    The function has no return
    """
    select_groups = []
    if len(namelist) == 0:
        select_groups = optimizer.param_groups
    else:
        for param_group in optimizer.param_groups:
            if param_group["name"] in namelist:
                select_groups.append(param_group)

    for param_group in select_groups:
        if mode == "auto":
            p = float(epoch) / args.epochs
            lr = args.base_lr / ((1 + 10 * p) ** 0.75)
            lr_pretrain = lr * value
            for param_group in optimizer.param_groups:
                if param_group["name"] == "pre-trained":
                    param_group["lr"] = lr_pretrain
                else:
                    param_group["lr"] = lr
        elif mode == "rel":
            param_group["lr"] = param_group["lr"] * value
        elif mode == "abs":
            param_group["lr"] = value
