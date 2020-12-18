# Copyright (c) Gorilla-Lab. All rights reserved.
import inspect
import warnings
from functools import partial
from typing import Optional, Dict, Type


class Registry:
    r"""A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     "(name={}, items={})".format(self._name, self._module_dict)
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        r"""Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, "
                            "but got {}".format(type(module_class)))

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError("{} is already registered "
                           "in {}".format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "The old API of register_module(module, force=False) "
            "is deprecated and will be removed, please use the new API "
            "register_module(name=None, force=False, module=None) instead.")
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
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
            raise TypeError("force must be a boolean, but got {}".format(type(force)))
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError("name must be a str, but got {}".format(type(name)))

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register


def auto_registry(registry: Registry, cls_dict: Dict):
    r"""Author: liang.zhihao

    Args:
        registry (Registry): Registry
        cls_dict (Dict): dict of Class
    """
    for key, cls in cls_dict.items():
        # skip the "_" begin
        if key.startswith("_"):
            continue
        # skip function(just register Class)
        if not isinstance(cls, Type):
            continue
        registry._register_module(cls)


def build_from_cfg(cfg: Dict, registry: Registry, default_args: Optional[Dict]=None):
    r"""Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "name".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict, but got {}".format(type(cfg)))
    if "name" not in cfg:
        if default_args is None or "name" not in default_args:
            raise KeyError(
                "`cfg` or `default_args` must contain the key 'type', "
                "but got {}\n{}".format(cfg, default_args))
    if not isinstance(registry, Registry):
        raise TypeError("registry must be an mmcv.Registry object, "
                        "but got {}".format(type(registry)))
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError("default_args must be a dict or None, "
                        "but got {}".format(type(default_args)))

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_name = args.pop("name")
    if isinstance(obj_name, str):
        obj_cls = registry.get(obj_name)
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_name, registry.name))
    elif inspect.isclass(obj_name):
        obj_cls = obj_name
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_name)))

    return obj_cls(**args)

