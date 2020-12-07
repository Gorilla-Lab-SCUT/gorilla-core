# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict

from torch.nn.utils import clip_grad

from ..config import Config

class GradClipper:
    def __init__(self, grad_clip_cfg: [Dict, Config]):
        name = grad_clip_cfg.pop("name")
        assert name in ["norm", "value"]
        self.grad_clip_cfg = grad_clip_cfg
        self.clipper = getattr(clip_grad, "clip_grad_{}_".format(name))

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return self.clipper(params, **self.grad_clip_cfg)


def build_grad_clipper(grad_clip_cfg):
    assert "name" in grad_clip_cfg
    return GradClipper(grad_clip_cfg)


