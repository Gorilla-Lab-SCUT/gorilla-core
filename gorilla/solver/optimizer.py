# Copyright (c) Facebook, Inc. and its affiliates.
# auto registry all inplace optimizetr
from torch.optim import *
import torch.optim.optimizer as optimizer

try:
    # a rich pytorch optimizer library
    # https://github.com/jettify/pytorch-optimizer
    from torch_optimizer import *
except:
    pass

from gorilla.core import OPTIMIZERS, auto_registry
auto_registry(OPTIMIZERS, globals(), optimizer.Optimizer)
