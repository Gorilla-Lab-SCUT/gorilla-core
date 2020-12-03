# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module

import numpy as np
import torch
import torchvision
from torch.optim import Optimizer
from torch.utils import model_zoo
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler

from ..core import get_dist_info


def is_module_wrapper(module):
    """Check if a module is a module wrapper.
    The following modules are regarded as
    module wrappers: DataParallel, DistributedDataParallel
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module"s
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if "num_batches_tracked" not in key
    ]

    if unexpected_keys:
        err_msg.append("unexpected key in source state_dict: {}\n".format(", ".join(unexpected_keys)))
    if missing_keys:
        err_msg.append("missing keys in source state_dict: {}\n".format(", ".join(missing_keys)))

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    r"""Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "No state_dict found in checkpoint file {}".format(filename))
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def resume(model,
           filename,
           optimizer=None,
           scheduler=None,
           resume_optimizer=True,
           resume_scheduler=True,
           map_location="default"):
    if map_location == "default":
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                model,
                filename,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(model, filename)
    else:
        checkpoint = load_checkpoint(model, filename, map_location=map_location)

    if "optimizer" in checkpoint and resume_optimizer:
        if optimizer is None:
            warnings.warn("optimizer is None, skip the optimizer loading")
        elif isinstance(optimizer, Optimizer):
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif isinstance(optimizer, dict):
            for k in optimizer.keys():
                optimizer[k].load_state_dict(
                    checkpoint["optimizer"][k])
        else:
            raise TypeError(
                "Optimizer should be dict or torch.optim.Optimizer but got {}".format(type(optimizer)))

    if "scheduler" in checkpoint and resume_scheduler:
        if scheduler is None:
            warnings.warn("scheduler is None, skip the scheduler loading")
        elif isinstance(scheduler, _LRScheduler):
            scheduler.load_state_dict(checkpoint['scheduler'])
        elif isinstance(scheduler, dict):
            for k in scheduler.keys():
                scheduler[k].load_state_dict(
                    checkpoint["scheduler"][k])
        else:
            raise TypeError(
                "scheduler should be dict or torch.optim.lr_scheduler._LRScheduler but got {}".format(type(scheduler)))



def save_checkpoint(model, filename, optimizer=None, scheduler=None, meta=None):
    r"""Save checkpoint to file.
    The checkpoint will have 3 fields:
        ``meta``, ``state_dict`` and ``optimizer``.
        By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got {}".format(type(meta)))
    meta.update(time=time.asctime())

    os.makedirs(osp.dirname(filename), exist_ok=True)
    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        "meta": meta,
        "state_dict": weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if optimizer is not None:
        if isinstance(optimizer, Optimizer):
            checkpoint["optimizer"] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint["optimizer"] = {}
            for name, optim in optimizer.items():
                checkpoint["optimizer"][name] = optim.state_dict()
        else:
            raise TypeError(
                "Optimizer should be dict or torch.optim.Optimizer but got {}".format(type(optimizer)))

    # save lr_scheduler state dict in the checkpoint
    if scheduler is not None:
        if isinstance(scheduler, _LRScheduler):
            checkpoint["scheduler"] = scheduler.state_dict()
        elif isinstance(scheduler, dict):
            checkpoint["scheduler"] = {}
            for name, sche in scheduler.items():
                checkpoint["scheduler"][name] = sche.state_dict()
        else:
            raise TypeError(
                "scheduler should be dict or torch.optim.lr_scheduler._LRScheduler but got {}".format(type(scheduler)))
        
    # immediately flush buffer
    with open(filename, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()


def load_url_dist(url, model_dir=None):
    r"""In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module("torchvision.models.{}".format(name))
        if hasattr(_zoo, "model_urls"):
            _urls = getattr(_zoo, "model_urls")
            model_urls.update(_urls)
    return model_urls


def _load_checkpoint(filename, map_location=None):
    r"""Load checkpoint from somewhere (modelzoo, file, url).
    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.
    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith("modelzoo://"):
        warnings.warn("The URL scheme of 'modelzoo://' is deprecated, please "
                      "use 'torchvision://' instead")
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith("torchvision://"):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith(("http://", "https://")):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError("{} is not a checkpoint file".format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def weights_to_cpu(state_dict):
    r"""Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    r"""Saves module state to `destination` dictionary.
    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix="", keep_vars=False):
    r"""Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.
    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def resume_checkpoint(model, cfg):
    if not osp.isfile(cfg.resume):
        raise ValueError('The file to be resumed is not existed', cfg.resume)

    print("==> loading checkpoints '{}'".format(cfg.resume))
    state = torch.load(cfg.resume)

    if cfg.method == "DANN":
        model.G_f.load_state_dict(state["G_f"])
        model.G_y.load_state_dict(state["G_y"])
        model.G_d.load_state_dict(state["G_d"])
    else:
        raise NotImplementedError("method: {}".format(cfg.method))

    return model


def save_summary(filepath, state, dir_save_file, best, desc="loss", smaller=True, overwrite=False):
    r"""
    Save a summary npy file, which contain path of the best model and its performance.
    Parameters
    ----------
    filepath: string
        Path of .npy file
    state: dict
        Including many infos, such as epoch, arch, model parameters, best_prec1, and optimizers
    dir_save_file: string
        Path to save variable 'state'
    best: float
        Best performance of the model, can be loss, accuracy or others.
    desc: string
        Description of the result, such as 'acc', 'loss' and so on
    smaller: bool (default: True)
        The indicator of whether the performance is the smaller the better (default) or the bigger the better
    Example
    ----------
    An example of the result .npy file:
    {'test/test.pth.tar': ['acc', 0.01],
     'test/test1.pth.tar': ['loss', 0.1],
     'test/test2.pth.tar': ['acc', 0.01]}
    """
    try:
        summary = np.load(filepath, allow_pickle=True).item()
    except: # the summary file has not been created
        summary = {}
    if overwrite:
        torch.save(state, dir_save_file)
        summary[dir_save_file] = [desc, best]
    else:
        if dir_save_file in summary.keys():
            if smaller:
                if best < summary[dir_save_file][1]: # refresh the old state info to a better one
                    torch.save(state, dir_save_file)
                    summary[dir_save_file] = [desc, best]
            else:
                if best > summary[dir_save_file][1]: # refresh the old state info to a better one
                    torch.save(state, dir_save_file)
                    summary[dir_save_file] = [desc, best]

        else: # new state
            torch.save(state, dir_save_file)
            summary[dir_save_file] = [desc, best]

    # from pprint import pprint
    # print(filepath)
    # pprint(summary)
    np.save(filepath, summary)
