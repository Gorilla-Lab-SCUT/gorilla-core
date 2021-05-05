# Copyright (c) Gorilla-Lab. All rights reserved.
import sys
import inspect
from typing import Optional, Dict, Type

from termcolor import colored


class Registry:
    def __init__(self, name: str):
        r"""A registry to map strings to classes.
        Args:
            name (str): Registry name.
        """
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key: str):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(type={colored(self._name, 'red')})\n"
        for key, value in self._module_dict.items():
            format_str += f"{colored(key, 'blue')}:\n"
            format_str += f"    {value}:\n"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> Type:
        r"""Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in self.name{self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self,
                        name: Optional[str]=None,
                        force: bool=False,
                        module: Optional[Type]=None):
        r"""Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry("backbone")
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry("backbone")
            >>> @backbones.register_module(name="mnet")
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry("backbone")
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be a str, but got {type(name)}")

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register


def auto_registry(registry: Registry,
                  cls_dict: Dict,
                  type=object,
                  force: bool=False) -> None:
    r"""Author: liang.zhihao

    Args:
        registry (Registry): Registry
        cls_dict (Dict): dict of Class
        type: typing to filter out
    """
    for key, cls in cls_dict.items():
        # skip the "_" begin
        if key.startswith("_"):
            continue
        # skip function(default register Class)
        if not isinstance(cls, Type):
            continue
        # keep the son class if type is define
        if not issubclass(cls, type):
            continue
        registry._register_module(cls, force=force)


def build_from_cfg(cfg: Dict,
                   registry: Registry,
                   default_args: Optional[Dict]=None) -> object:
    r"""Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                f"`cfg` or `default_args` must contain the key 'type', "
                f"but got {cfg}\n{default_args}")
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be an mmcv.Registry object, "
                        f"but got {type(registry)}")
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(f"default_args must be a dict or None, "
                        f"but got {type(default_args)}")

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f"type must be a str or valid type, but got {type(obj_type)}")

    return obj_cls(**args)


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type

    Args:
        info (dict): Object types and arguments
        parent (:class:`modules`):
        default_args (dict, optional):
    """
    assert isinstance(info, dict) and "type" in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)

