# Copyright (c) Gorilla-Lab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .weight_init import constant_init, kaiming_init
from .layer_builder import get_torch_layer_caller


class GorillaConv(nn.Sequential):
    # TODO: modify comment
    r"""A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon build map: `get_torch_layer_caller`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
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
        D: The convolutional dimension. Default: 2
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type="ReLU").
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ["zeros", "circular"] with official
            implementation and ["reflect"] with our own implementation.
            Default: "zeros".
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ("conv", "norm", "act").
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 stride: int=1,
                 padding: int=0,
                 dilation: int=1,
                 groups: int=1,
                 bias: bool=True,
                 name: str="",
                 D: int=2,
                 norm_cfg: Optional[Dict]=dict(type="BN2d"),
                 act_cfg: Optional[Dict]=dict(type="ReLU", inplace=True),
                 with_spectral_norm: bool=False,
                 padding_mode: List="zeros",
                 order: List[str]=["conv", "norm", "act"]):
        super().__init__()
        assert D in [1, 2, 3]
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert padding_mode in ["zeros", "reflect", "replicate", "circular"]

        assert set(order).difference(set(["conv", "norm", "act"])) == set()
        # modify a func's default params will affect next call, so it is
        # necessary to use deepcopy if you want to modify a dafault params
        # in the func
        self.order = deepcopy(order)
        self.act_cfg = deepcopy(act_cfg)
        self.norm_cfg = deepcopy(norm_cfg)

        # if the conv layer is before a norm layer, bias is unnecessary.\
        with_norm = (self.norm_cfg is not None)
        if with_norm:
            bias = False

        # reset padding to 0 for conv module
        conv_padding = padding
        # build convolutional layer
        conv_caller = get_torch_layer_caller(f"Conv{D}d")
        conv = conv_caller(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=conv_padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias,
                           padding_mode=padding_mode)

        if with_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        # build normalization layers
        norm = None
        if with_norm:
            if self.order.index("norm") > self.order.index("conv"):
                num_features = conv.out_channels
            else:
                num_features = conv.in_channels
            self.norm_cfg.update(num_features=num_features)
            norm_caller = get_torch_layer_caller(self.norm_cfg.pop("type"))
            norm = norm_caller(**self.norm_cfg)
        else:
            if "norm" in self.order:
                self.order.remove("norm")

        # Use msra init by default
        self.init_weights(conv, norm)

        # build activation layer
        with_act = (self.act_cfg is not None)
        act = None
        if with_act:
            act_caller = get_torch_layer_caller(self.act_cfg.pop("type"))
            act = act_caller(**self.act_cfg)
        else:
            if "act" in self.order:
                self.order.remove("act")

        # build layer according to the order
        for layer in self.order:
            self.add_module(name + layer, eval(layer))

    def init_weights(self, conv, norm):
        # TODO: modify this
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's conv layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(conv, "init_weights"):
            a, nonlinearity = 0, "relu"
            if self.act_cfg is not None and self.act_cfg["type"] == "LeakyReLU":
                a = self.act_cfg.get("negative_slope", 0.01)
                nonlinearity = "leaky_relu"
            kaiming_init(conv, a=a, nonlinearity=nonlinearity)
        if norm is not None:
            constant_init(norm, 1, bias=0)
