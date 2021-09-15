# Copyright (c) Gorilla-Lab. All rights reserved.
import sys
import inspect
from functools import wraps
from typing import Callable, Optional, Dict, Type
from typing import Any, Dict, Iterable, Iterator, Tuple

from tabulate import tabulate
from termcolor import colored


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any, force: bool = False) -> None:

        if not force and name in self._obj_map:
            raise KeyError(f"An object named '{name}' was already "
                           f"registered in '{self._name}' registry!\n"
                           f"Or let `force=True` to cover the origin")
        self._obj_map[name] = obj

    def register(self, obj: Any = None, force: bool = False) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        Example:
            >>> backbones = Registry("backbone")
            >>> @backbones.register()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry("backbone")
            >>> @backbones.register(name="mnet")
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry("backbone")
            >>> class ResNet:
            >>>     pass
            >>> backbones.register(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
        """
        if obj is None:
            # used as a decorator
            @wraps(obj)
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class, force)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, force)

    # NOTE: to fix the `register_module` API
    def register_module(self, obj: Any = None, force: bool = False):
        return self.register(obj, force)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' "
                           f"found in '{self._name}' registry!")
        return ret

    @property
    def name(self) -> str:
        return self._name

    @property
    def obj_map(self) -> Dict:
        return self._obj_map

    def __len__(self):
        return len(self._obj_map)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = [colored("Names", "red"), colored("Objects", "green")]
        table = tabulate(
            ((colored(k, "blue"), v) for k, v in self._obj_map.items()),
            headers=table_headers,
            tablefmt="fancy_grid")
        return f"Registry of {self._name}:\n" + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


def auto_registry(registry: Registry,
                  cls_dict: Dict,
                  type: Type = object,
                  force: bool = False) -> None:
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
        registry.register(cls, force=force)


def build_from_cfg(cfg: Dict,
                   registry: Registry,
                   default_args: Optional[Dict] = None) -> object:
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
            raise KeyError("`cfg` or `default_args` must contain the key "
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
                  parent: Optional[object] = None,
                  default_args: Optional[Dict] = None):
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
        raise TypeError(f"type must be callable, but got {type(obj_type)}")

    if default_args is not None:
        for name, value in default_args.items():
            kwargs.setdefault(name, value)
    return obj_type(**kwargs)
