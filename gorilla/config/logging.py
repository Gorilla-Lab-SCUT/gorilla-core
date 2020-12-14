# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import logging
from collections import OrderedDict

import torch.distributed as dist

from ..utils import timestamp
from ..core import convert_list_str

logger_initialized = {}


def collect_logger(root: str="log",
                   prefix: str=None,
                   suffix: str=None,
                   **kwargs) -> [str, logging.Logger]:
    r"""Author: liang.zhihao
    A easy combination of get_log_dir and get_logger, use the timestamp
    as log file's name

    Args:
        root (str, optional): the root directory of logger. Defaults to "log".
        prefix (str, optional): the extra prefix. Defaults to None.
        suffix (str, optional): the extra suffix. Defaults to None.

    Returns:
        [str, logging.Logger]: the log dir and the logger
    """
    log_dir = get_log_dir(root,
                          prefix,
                          suffix,
                          **kwargs)
    
    time_stamp = timestamp()
    log_file = osp.join(log_dir, "{}.log".format(time_stamp))
    logger = get_logger(log_file, timestamp=time_stamp)

    return log_dir, logger


def get_log_dir(root: str="log",
                prefix: str=None,
                suffix: str=None,
                **kwargs) -> str:
    r"""Author: liang.zhihao
    Get log dir according to the given params key-value pair

    Args:
        root (str, optional): the root directory of logger. Defaults to "log".
        prefix (str, optional): the extra prefix. Defaults to None.
        suffix (str, optional): the extra suffix. Defaults to None.

    Example:
        >>> import gorilla
        >>> # dynamic concatenate
        >>> gorilla.get_log_dir(lr=0.001, bs=4)
        "log/lr_0.001_bs_4"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam")
        "log/lr_0.001_bs_4_Adam"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", suffix="test") # add the suffix
        "log/lr_0.001_bs_4_Adam_test"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", prefix="new") # add the prefix
        "log/new_lr_0.001_bs_4_Adam_test"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", prefix="new", suffix="test") # add both prefix and suffix 
        "log/lr_0.001_bs_4_Adam_test"

    Returns:
        str: the directory path of log
    """
    # concatenate the given parameters
    args_dict = OrderedDict(kwargs)
    params = []
    for key, value in args_dict.items():
        params.extend([key, value])
    
    # deal with prefix
    if prefix is not None:
        params.insert(0, prefix)
    
    # deal with suffix
    if suffix is not None:
        params.append(suffix)
    
    # convert all parameters as str
    params = convert_list_str(params)

    # get log dir and make
    sub_log_dir = "_".join(params)
    log_dir = osp.join(root, sub_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def get_logger(log_file=None, name="gorilla", log_level=logging.INFO, timestamp=None):
    r"""Initialize and get a logger by name.
        If the logger has not been initialized, this method will initialize the
        logger by adding one or two handlers, otherwise the initialized logger will
        be directly returned. During initialization, a StreamHandler will always be
        added. If `log_file` is specified and the process rank is 0, a FileHandler
        will also be added.

    Args:
        name (str): Logger name. Default to 'gorilla'
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        timestamp (str, optional): The timestamp of logger.
            Defaults to None
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.timestamp = timestamp
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        if not osp.isdir(osp.dirname(log_file)):
            os.makedirs(osp.dirname(log_file))
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    r"""Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_logger(log_file)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            "'silent' or None, but got {}".format(type(logger)))

