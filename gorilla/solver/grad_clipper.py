# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict

from torch.nn.utils import clip_grad

class GradClipper:
    def __init__(self, grad_clip=None, clip_type=None):
        assert grad_clip is None or isinstance(grad_clip, Dict)
        assert clip_type is None or clip_type in ["norm", "value"]
        self.grad_clip = grad_clip
        self.clipper = getattr(clip_grad, "clip_grad_{}_".format(clip_type))

    def clip_grads(self, params):
        assert self.grad_clip is not None
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return self.clipper(params, **self.grad_clip)


