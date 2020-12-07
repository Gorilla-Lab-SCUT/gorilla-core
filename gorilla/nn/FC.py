# Copyright (c) Gorilla-Lab. All rights reserved.
from copy import deepcopy
import warnings
import math

import torch
import torch.nn as nn

from .weight_init import constant_init, kaiming_init
from .layer_builder import get_torch_layer_caller


class GorillaFC(nn.Sequential):
    # TODO: modify comment and test
    r"""A FC block that bundles FC/norm/activation layers.

    This block simplifies the usage of fully connect layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon build method: "get_torch_layer_caller"

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the fully connect layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        FC_cfg (dict): Config dict for fully connect layer. Default: None
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type="ReLU").
        order (tuple[str]): The order of FC/norm/activation layers. It is a
            sequence of "FC", "norm" and "act". Common examples are
            ("FC", "norm", "act") and ("act", "FC", "norm").
            Default: ("FC", "norm", "act").
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 name="",
                 norm_cfg=dict(name="BN1d"),
                 act_cfg=dict(name="ReLU", inplace=True),
                 dropout=None,
                 order=["FC", "norm", "act", "dropout"]):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)

        assert set(order).difference(set(["FC", "norm", "act",
                                          "dropout"])) == set()

        self.order = deepcopy(order)
        self.norm_cfg = deepcopy(norm_cfg)
        self.act_cfg = deepcopy(act_cfg)

        # if the FC layer is before a norm layer, bias is unnecessary.\
        with_norm = (self.norm_cfg is not None)
        # if with_norm:
        #     bias = False

        # build FC layer
        FC = nn.Linear(in_features, out_features, bias)

        # build normalization layers
        norm = None
        if with_norm:
            if self.order.index("norm") > self.order.index("FC"):
                num_features = FC.out_features
            else:
                num_features = FC.in_features
            self.norm_cfg.update(num_features=num_features)
            norm_caller = get_torch_layer_caller(self.norm_cfg.pop("name"))
            norm = norm_caller(**self.norm_cfg)
        else:
            if "norm" in self.order:
                self.order.remove("norm")

        # Use msra init by default
        self.init_weights(FC, norm)

        # build activation layer
        with_act = (self.act_cfg is not None)
        act = None
        if with_act:
            act_caller = get_torch_layer_caller(self.act_cfg.pop("name"))
            act = act_caller(**self.act_cfg)
        else:
            if "act" in self.order:
                self.order.remove("act")

        # build dropout layer
        with_dropout = (dropout is not None)
        if with_dropout:
            dropout = nn.Dropout(p=dropout)
        else:
            if "dropout" in self.order:
                self.order.remove("dropout")

        for layer in self.order:
            self.add_module(name + layer, eval(layer))

    def init_weights(self, FC, norm):
        # TODO: modify this
        # 1. It is mainly for customized fully connect layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized fully connect layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's fully connect layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(FC, "init_weights"):
            a, nonlinearity = math.sqrt(5), "relu"
            if self.act_cfg is not None and self.act_cfg["name"] == "LeakyReLU":
                a = self.act_cfg.get("negative_slope", 0.01)
                nonlinearity = "leaky_relu"
            kaiming_init(FC,
                         a=a,
                         mode="fan_in",
                         nonlinearity="leaky_relu",
                         distribution="uniform")
        if norm is not None:
            constant_init(norm, 1, bias=0)


class MultiFC(nn.Module):
    r"""Multi FC layers based on GorillaFC.
    Args:
        params (list): each element is a params dict about one GorillaFC layer
    """
    def __init__(self, params):
        super().__init__()
        layers = []
        for param in params:
            layers.append(GorillaFC(**param))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
