# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import torch.nn as nn


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode="fan_out",
                 nonlinearity="relu",
                 bias=0,
                 distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight,
                                 a=a,
                                 mode=mode,
                                 nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight,
                                a=a,
                                mode=mode,
                                nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def c2_xavier_init(module: nn.Module) -> None:
    r"""
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def c2_msra_init(module: nn.Module) -> None:
    r"""
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def bias_init_with_prob(prior_prob):
    r"""initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
