# Copyright (c) Gorilla-Lab. All rights reserved.
from abc import ABCMeta
from termcolor import colored
from typing import List, Union

from ..base_solver import BaseSolver
from ...core import build_hook

class Hook(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.priority = 50 # NORMAL


class HookManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._hooks = []
        self._status = ["before_epoch", "before_step", "after_step", "after_epoch"]

    def register_hook(self, hooks: Union[List[Hook], Hook]):
        r"""
        Register hooks work for solver. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Hook], Hook): list of hooks
        """
        if isinstance(hooks, Hook):
            _hooks = [hooks]
        else:
            _hooks = []
            for hook in hooks:
                if hook is not None and isinstance(Hook, hook):
                    _hooks.append(hook)
            
        self._hooks.extend(_hooks)

    def concat_solver(self, solver:BaseSolver):
        self.solver = solver

    def register_hook_from_cfg(self, hook_cfg):
        r"""Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys "type"
              and "priority" indicating its type and priority.

        Notes:
            The specific hook class to register should not use "type" and
            "priority" arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        hook = build_hook(hook_cfg)
        self.register_hook(hook)

    def call_hook(self, fn_name):
        r"""Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_epoch".
        """
        for hook in self._hooks:
            # check fn auto
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

    @property
    def hooks(self):
        return self._hooks

    @property
    def status(self):
        return self._status

    def __repr__(self) -> str:
        r"""Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_epoch".
        """
        content = [""]
        content.append(colored(" pipeline ".center(50, "*"), "yellow"))
        for status in self.status:
            content.append(colored(status.center(50, "="), "blue"))
            empty_flag = True
            for hook in self.hooks:
                if not hasattr(hook, status):
                    continue
                content.append(colored(hook.__class__.__name__ + ": ", "red"))
                content.append(" " * 4 + getattr(hook, status).__doc__)
                empty_flag = False
            if empty_flag:
                content.append(colored("Empty", "green"))
            content.append(colored("="*50, "blue"))
            content.append("")
        content.append(colored("".center(50, "*"), "yellow"))

        return "\n".join(content)

