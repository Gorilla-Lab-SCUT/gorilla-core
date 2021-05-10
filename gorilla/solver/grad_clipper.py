# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict, Generator, Optional, Union

from torch.nn import Module
from torch.nn.utils import clip_grad

from ..config import Config

class GradClipper:
    def __init__(self, grad_clip_cfg: Optional[Union[Dict, Config]]=None):
        self.clip = True
        if grad_clip_cfg is not None:
            try:
                clip_type = grad_clip_cfg.pop("type")
            except:
                clip_type = "norm"

            assert clip_type in ["norm", "value"]
            self.grad_clip_cfg = grad_clip_cfg
            self.clipper = getattr(clip_grad, f"clip_grad_{clip_type}_")
        else:
            self.clip = False

    def clip_grads(self, params: Union[Module, Generator]):
        if self.clip:
            # extract parameters for module
            if isinstance(params, Module):
                params = params.parameters()
            params = list(
                filter(lambda p: p.requires_grad and p.grad is not None, params))
            if len(params) > 0:
                self.clipper(params, **self.grad_clip_cfg)


def build_grad_clipper(grad_clip_cfg):
    assert "type" in grad_clip_cfg
    return GradClipper(grad_clip_cfg)


