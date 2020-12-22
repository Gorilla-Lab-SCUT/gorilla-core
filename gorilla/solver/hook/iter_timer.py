# Copyright (c) Open-MMLab. All rights reserved.
import time

from gorilla.core import HOOKS
from .hook import Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, solver):
        r"""init start time for record consume time"""
        self.t = time.time()

    def before_iter(self, solver):
        r"""calculate and show the time cost of preparing data"""
        solver.log_buffer.update({"data_time": time.time() - self.t})

    def after_iter(self, solver):
        r"""calculate and show the time cost of an iter"""
        solver.log_buffer.update({"time": time.time() - self.t})
        self.t = time.time()
