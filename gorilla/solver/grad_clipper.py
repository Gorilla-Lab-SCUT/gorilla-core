# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict

from torch.nn.utils import clip_grad

from ..config import Config

class GradClipper:
    def __init__(self, grad_clip_cfg: [Dict, Config]):
        clip_type = grad_clip_cfg.pop("type")
        assert clip_type in ["norm", "value"]
        self.grad_clip_cfg = grad_clip_cfg
        self.clipper = getattr(clip_grad, f"clip_grad_{clip_type}_")

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return self.clipper(params, **self.grad_clip_cfg)


def build_grad_clipper(grad_clip_cfg):
    assert "type" in grad_clip_cfg
    return GradClipper(grad_clip_cfg)


