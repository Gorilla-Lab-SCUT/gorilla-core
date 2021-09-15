# Copyright (c) Open-MMLab. All rights reserved.
import os
import json
import tempfile
import warnings
from typing import Optional
from argparse import Namespace

from addict import Dict

from ..utils import check_file

BASE_KEY = "_base_"
RESERVED_KEYS = ["filename", "text"]


class ConfigDict(Dict):
    r"""ConfigDict based on Dict, which use to convert the config
        file into config dict
    """
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"`{self.__class__.__name__}` object has no attribute `{name}`"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config(object):
    r"""A facility for config and config files.
        It supports common file formats as configs: python/json/yaml. The interface
        is the same as a dict object and also allows access config values as
        attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {"b1": [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile("./configs/test.py")
        >>> cfg.filename
        "/home/gorilla_lab/code/gorilla/configs/test.py"
        >>> cfg.item4
        "test"
        >>> cfg
        "Config [path: /home/gorilla_lab/code/gorilla/configs/test.py]: "
        "{"item1": [1, 2], "item2": {"a": 0}, "item3": True, "item4": "test"}"
    """
    def __init__(self,
                 cfg_dict: Optional[Dict] = None,
                 cfg_text: Optional[str] = None,
                 filename: Optional[str] = None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, "
                            f"but got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Config, self).__setattr__("_text", text)


    @staticmethod
    def _file2dict(filename: str):
        filename = os.path.abspath(os.path.expanduser(filename))
        check_file(filename)
        from gorilla.fileio import load
        cfg_dict = ConfigDict(load(filename))

        with open(filename, "r") as f:
            cfg_text = f.read()

        # here cfg_dict is still the same as content in --config file,
        # and the code block below read 4 sub-config file then merge into one.
        if BASE_KEY in cfg_dict:
            cfg_dir = os.path.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(os.path.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    # e.g. sub-config file about dataset should not overlap with
                    # the one about model
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        r"""merge dict ``a`` into dict ``b`` (non-inplace).
        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.
        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {"obj": {"a": 2}}
        """
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                allowed_types = dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config.")
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename: str):
        r"""cfg_text is the text content read from 5 files, and cfg_dict is
            a dict resolved by the text content.
        """
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.
        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!
        Returns:
            obj:`Config`: Config obj.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn(
                "Please check 'file_format', the file format may be .py")

        with tempfile.NamedTemporaryFile("w", suffix=file_format) as temp_file:
            temp_file.write(cfg_str)
            temp_file.flush()
            cfg = Config.fromfile(temp_file.name)
        return cfg

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def text(self) -> str:
        return self._text

    def __repr__(self) -> str:
        content = f"Config (path: {self.filename})\n"
        content += json.dumps(self._cfg_dict, indent=4, ensure_ascii=False)
        return content

    def __len__(self) -> int:
        return len(self._cfg_dict)

    def __getattr__(self, name: str):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name: str, value: Dict):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Dict):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, file: Optional[str] = None, **kwargs):
        cfg_dict = self._cfg_dict.to_dict()
        from gorilla.fileio import dump
        if file is None:
            # output the content
            file_format = self.filename.split(".")[-1]
            if file_format == "py":
                return self.text
            else:
                return dump(cfg_dict, file_format=file_format, **kwargs)
        else:
            if file.endswith("py"):
                with open(file, "w") as f:
                    f.write(self.text)
            else:
                dump(cfg_dict, file, **kwargs)

    def merge_from_dict(self, options: Dict):
        r"""Merge list into cfg_dict.
        Merge the dict parsed by MultipleKVAction into this cfg.
        Examples:
            >>> options = {"model.backbone.depth": 50,
            ...            "model.backbone.with_cp":True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type="ResNet"))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))
            
            # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type="LoadImage"), dict(type="LoadAnnotations")]))
            >>> options = dict(pipeline={"0": dict(type="SelfLoadImage")})
        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            if v is None:  # handle the case when a parameter simultaneously appears in argparse and config file
                continue
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = self._cfg_dict
        cfg_dict = Config._merge_a_into_b(option_cfg_dict, cfg_dict)
        # NOTE: strange phenomenon
        # self._cfg_dict = cfg_dict
        super(Config, self).__setattr__("_cfg_dict", cfg_dict)


def merge_cfg_and_args(cfg: Optional[Config] = None,
                       args: Optional[Namespace] = None) -> Config:
    r"""merge args and cfg into a Config by calling 'merge_from_dict' func

    Args:
        cfg (Config, optional): Config from cfg file.
        args (Namespace, optional): Argument parameters input.

    Returns:
        Config: Merged Config
    """
    assert cfg is not None or args is not None, "'cfg' or 'args' can not be None simultaneously"

    if cfg is None:
        cfg = Config()
    else:
        assert isinstance(
            cfg, Config
        ), f"'cfg' must be None or gorilla.Config, but got {type(cfg)}"
    if args is None:
        args = Namespace()
    else:
        assert isinstance(
            args, Namespace
        ), f"'args' must be None or argsparse.Namespace, but got {type(args)}"

    # convert namespace into dict
    args_dict = vars(args)
    cfg.merge_from_dict(args_dict)
    return cfg

