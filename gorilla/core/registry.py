# Copyright (c) Gorilla-Lab. All rights reserved.
import sys
import inspect
import warnings
from functools import partial
from typing import Callable, List, Optional, Dict, Type, Union

from termcolor import colored

from .misc import is_seq_of


class Registry:
    def __init__(self,
                 name: str,
                 build_func: Optional[Callable]=None,
                 parent: Optional[object]=None,
                 scope: Optional[str]=None):
        r"""A registry to map strings to classes.
        
        Registered object could be built from registry.
        Example:
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = MODELS.build(dict(type='ResNet'))
        # TODO: fix comment
        Please refer to https://mmcv.readthedocs.io/en/latest/registry.html for
        advanced useage.

        Args:
            name (str): Registry name.
            build_func(func, optional): Build function to construct instance from
                Registry, func:`build_from_cfg` is used if neither ``parent`` or
                ``build_func`` is specified. If ``parent`` is specified and
                ``build_func`` is not given,  ``build_func`` will be inherited
                from ``parent``. Default: None.
            parent (Registry, optional): Parent registry. The class registered in
                children registry could be built from parent. Default: None.
            scope (str, optional): The scope of registry. It is the key to search
                for children registry. If not specified, scope will be the name of
                the package where class is defined, e.g. gorilla2d, gorilla3d.
                Default: None.
        """
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # assert `parent`'s type if `parent` is given
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

        # `self.build_func` will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key: str):
        return self.get(key) is not None

    def __repr__(self):
        # # mmcv style
        # format_str = self.__class__.__name__ + \
        #              f'(name={self._name}, ' \
        #              f'items={self._module_dict})'

        # color rich style
        format_str = self.__class__.__name__ + f"(type={colored(self._name, 'red')})\n"
        for key, value in self._module_dict.items():
            format_str += f"{colored(key, 'blue')}:\n"
            format_str += f"    {value}\n"
        return format_str

    @staticmethod
    def infer_scope():
        r"""Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.
        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key: str):
        r"""Split scope and key.
        The first scope will be split from key.
        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def _add_children(self, registry):
        """Add children for a registry.
        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.
        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def get(self, key: str) -> Type:
        r"""Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope: # default
            # get from self
            return self._module_dict.get(real_key, None)
        else:
            # get from `self._children`
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)
    
    def _register_module(self,
                         module_class: Type,
                         module_name: Optional[Union[str, List[str]]]=None,
                         force: bool=False):
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")

        # get the name of module, default to module's `__name__`
        if module_name is None:
            module_name = module_class.__name__
        # wrap by list
        if isinstance(module_name, str):
            module_name = [module_name]
        # register with statement
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered "
                               f"in {self.name}")
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn("The old API of register_module(module, force=False) "
                      "is deprecated and will be removed, please use the new API "
                      "register_module(name=None, force=False, module=None) instead.")
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    # TODO: update the registery refer to mmcv
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
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(f"name must be either of None, an instance of "
                            f"str or a sequence of str, but got {type(name)}")

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register


def auto_registry(registry: Registry,
                  cls_dict: Dict,
                  type: Type=object,
                  force: bool=False) -> None:
    r"""Author: liang.zhihao

    Args:
        registry (Registry): Registry
        cls_dict (Dict): dict of Class
        type (Type): typing to filter out
        force (bool, optional):
            Whether to override an existing class with
            the same name for all class. Default to False
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
                "`cfg` or `default_args` must contain the key "
                f"'type', but got {cfg}\n{default_args}")
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
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


# NOTE: add example
def obj_from_dict(info: Dict,
                  parent: Optional[object]=None,
                  default_args: Optional[Dict]=None):
    r"""Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type

    Args:
        info (dict): Object types and arguments
        parent (:class:`modules`):
        default_args (dict, optional):
    """
    assert isinstance(info, dict) and "type" in info
    assert isinstance(default_args, dict) or default_args is None
    kwargs = info.copy()
    module_type = kwargs.pop("type")
    if isinstance(module_type, str):
        obj_type = module_type.split(".")[-1]
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            module = ".".join(module_type.split(".")[:-1])
            module = sys.modules[module]
            obj_type = getattr(module, obj_type)
    
    if not isinstance(obj_type, Callable):
        raise TypeError(
            "type must be callable, but got {}".format(type(obj_type))
        )

    if default_args is not None:
        for name, value in default_args.items():
            kwargs.setdefault(name, value)
    return obj_type(**kwargs)

