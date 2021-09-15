# Copyright (c) Open-MMLab. All rights reserved.
from gorilla.core import HOOKS
from .hook import Hook


@HOOKS.register()
class CallbackHook(Hook):
    def __init__(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)
