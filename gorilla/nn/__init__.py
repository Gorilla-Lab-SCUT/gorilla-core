# Copyright (c) Gorilla-Lab. All rights reserved.
from .weight_init import (bias_init_with_prob, constant_init, kaiming_init,
                          normal_init, uniform_init, xavier_init, c2_msra_init,
                          c2_xavier_init, geometric_init)

from .conv import GorillaConv
from .fc import GorillaFC, MultiFC, DenseFC
from .gcn import GraphConvolution, GCN
from .transformer import (Transformer, TransformerEncoder, TransformerDecoder,
                          TransformerEncoderLayer, TransformerDecoderLayer,
                          build_transformer)

from .vgg import VGG
from .alexnet import AlexNet
from .resnet import (BasicBlock, Bottleneck, resnet,
                     resnet18, resnet34, resnet50, resnet101, resnet152,
                     resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2)

# auto registry all defined module
from ..core import MODULES, auto_registry
import torch
auto_registry(MODULES, globals(), type=torch.nn.Module)

from .layer_builder import NAME_MAP, build_from_package, get_torch_layer_caller

__all__ = [k for k in globals().keys() if not k.startswith("_")]
