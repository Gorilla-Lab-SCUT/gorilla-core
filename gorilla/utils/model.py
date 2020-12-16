# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import pprint
import errno
import hashlib
import shutil
import re
import sys
import tempfile
import warnings
import inspect
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import numpy as np
from urllib.request import urlopen
from urllib.parse import urlparse
from tqdm import tqdm

from .debug import display

"""
directory
---------
classes

functions

load_state_dict_from_url: Loads the Torch serialized object at the given URL
check_model: Check if the demension of all modules is matching
check_params: Check parameters in models (support one or two models simultaneously)
check_grad: Check gradient of model parameters (support one or two models simultaneously)

"""
# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

def _get_torch_home():
    torch_home = osp.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  osp.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = osp.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = osp.basename(parts.path)
    cached_file = osp.join(model_dir, filename)
    if not osp.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)


def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = osp.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if osp.exists(f.name):
            os.remove(f.name)


def check_model(input_size, model, layer_wise=False, keep_hook=False):
    r"""Use an all-one input tensor to flow through the entire network, to check
        if the demension of all modules is matching (and if the params is loaded correctly)
        Beacuse if there exist fc layer in the model, the H and W of test input is unique, and 
        the in_channels of fc layer is hard to obtain, so give up the idea to automatically generate
        a test input, but use a external input instead.
        Note: If you want to compare the hierarchy of two network, directly
        print(model) may be a nice choice.
    Args:
        input_size (list | tuple): The size of input tensor, which is [C, H, W].
        model (nn.Module): The model need to be checked.
        layer_wise (bool, optional): Whether print layer-wise statistics.
        keep_hook (bool, optional): Only used in debugging, it will keep printing
            model input and output statistics after check_model() is executed.
    Returns:
        summary: OrderedDict, optional(disabled now)
            Contain informations of base modules of the model, each module has info about 
            input_shape, output_shape, num_classes, and trainable
    Usage:
        model = Model(args)
        check_model(input_size, model)
    """
    def get_sth(output, sth):
        if isinstance(output, tuple): # if the model has more than one output, "output" here will be a tuple
            result = {}
            for i, _ in enumerate(output):
                result[i] = OrderedDict()
                result[i] = get_sth(output[i], sth)
        elif sth == "sum":
            result = output.sum().item()
        elif sth == "max":
            result = output.max().item()
        elif sth == "min":
            result = output.min().item()
        elif sth == "mean":
            result = output.mean().item()
        elif sth == "size":
            result = list(output.size())
        return result

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            allow_base_modules_only = True # it control whether create summary for those middle modules
            if allow_base_modules_only:
                # add base module name if needed
                base_classes = ["Linear",
                                "Conv2d",
                                "Flatten",
                                "ReLU",
                                "PReLU",
                                "Sigmoid",
                                "Dropout",
                                "BatchNorm1d",
                                "BatchNorm2d",
                                "MaxPool2d",
                                "AdaptiveAvgPool2d"]
                if class_name not in base_classes:
                    return
            class_idx = classes_idx.get(class_name)
            if class_idx is None:
                class_idx = 0
                classes_idx[class_name] = 1
            else:
                classes_idx[class_name] += 1

            m_key = "{}-{} ({})".format(class_name, class_idx+1, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size()) # input is a tuple whose first element is a tensor
            summary[m_key]["output_shape"] = get_sth(output, "size")
            if layer_wise: # more exact checking
                summary[m_key]["input_sum"] = get_sth(input[0], "sum")
                summary[m_key]["output_sum"] = get_sth(output, "sum")
                summary[m_key]["output_max"] = get_sth(output, "max")
                summary[m_key]["output_min"] = get_sth(output, "min")
                summary[m_key]["output_mean"] = get_sth(output, "mean")

            params = 0
            if hasattr(module, "weight"):
                params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
                if module.weight.requires_grad:
                    summary[m_key]["trainable"] = True
                else:
                    summary[m_key]["trainable"] = False
            #if hasattr(module, "bias"):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]["num_params"] = params # not take bias into consideration
            pprint.pprint({m_key: summary[m_key]})

        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not module == model: # make sure "module" is a base module, such conv, fc and so on
            hooks.append(module.register_forward_hook(hook)) # hooks is used to record added hook for removing them later

    model.eval()
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.ones(2, *in_size) for in_size in input_size]
    else:
        x = torch.ones(2, *input_size) # 1 is batch_size dimension to adapt the model's structure

    if next( model.parameters() ).is_cuda:
        x = x.cuda()

    # create properties
    summary = OrderedDict()
    classes_idx = {}
    hooks = []
    # register hook
    model.apply(register_hook) # 递归地去给每个网络组件挂挂钩（不只是conv, fc这种底层组件，上面的Sequential组件也会被访问到）
    # make a forward pass
    I = inspect.getfullargspec(model.forward)
    # for standard nn.Modules class: self.forward(x)
    with torch.no_grad():
        if I.varargs is None and I.varkw is None:
            output = model(x)
        else: # for gorilla's BaseModel class: self.forward(be_train=True, **kwargs)
            data = dict(img=x,
                        sample_metas=[])
            output = model.forward_train(data, be_train=True)

    # remove these hooks
    if keep_hook:
        pass
    else:
        for h in hooks:
            h.remove()
    # pprint.pprint(summary)
    def display(name, out, num=""):
        # more exact comparison
        # tmp = out.view(-1, 1)
        # print("output {} size: {}, sum: {}".format(i, out.size(), tmp[:10]))
        print("{}{}: size={}, sum={:.5f}".format(name, num, list(out.size()), out.sum().item()))

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            display("output ", out, num=i)
    elif isinstance(output, dict):
        for key, out in output.items():
            # filter out sample_meta
            if isinstance(out, torch.Tensor):
                display(key, out)
    else:
        display("output", output)
    print("Check done.")
    # return summary


def check_params(model1, model2="", key=".", detailed=False):
    """
    Single model version (model2 has passed nothing):
        Check parameters in models, especially used in weight-share model to verify that
        the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    Double model version:
        Check parameters in two weight-share models, to verify that the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    key (str): Filter key, only display parameters whose name contain this key.
    """
    if model2 == "":
        if detailed:
            params1 = model1.state_dict().items()
        else:
            params1 = model1.named_parameters()
        for name, param in params1:
            if key in name:
                # transfrom to float32 for some int variable to get "mean",
                # such as num_batches_tracked
                display(name, param.type(torch.float32))
        return
    if detailed:
        params1 = iter(model1.state_dict().items())
        params2 = iter(model2.state_dict().items())
    else:
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

    m1_next, m2_next = True, True
    while m1_next or m2_next:
        try:
            name, param = params1.__next__()
            if key in name:
                # display(name, param)
                display(name, param.type(torch.float32))
        except StopIteration:
            m1_next = False
        try:
            name, param = params2.__next__()
            if key in name:
                # display(name, param)
                display(name, param.type(torch.float32))

        except StopIteration:
            m2_next = False


def check_grad(model1, model2=""):
    """
    Single model version (model2 has passed nothing):
        Check gradient of model parameters, especially used in weight-share model to verify that
        the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    Double model version:
        Check gradient in two weight-share models, to verify that the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    """
    if model2 == "":
        try:
            # for the condition that model1 is not a module, but an image (in adversial-sample-like training)
            if "Tensor" in str(model1.__class__).split("'")[1]:
                display("tensor", model1)

            for name, param in model1.named_parameters():
                display("grad of " + name, param.grad)
        except AttributeError as e:
            raise AttributeError("{}. Maybe the parameter '{}' in model is not used, please have a check.".format(e, name))
        
        return

        # minimum = 10086
        # maximum = 0
        # simple_sum = 0
        # simple_count = 0
        # try:
        #     # for the condition that model1 is not a module, but an image (in adversial-sample-like training)
        #     if "Tensor" in str(model1.__class__).split("'")[1]:
        #         print("min:", model1.grad.abs().min(), "max:", model1.grad.abs().max(), "mean:", model1.grad.abs().mean())
        #         return model1.grad.abs().mean()   

        #     for name, param in model1.named_parameters():
        #         if param.grad.abs().mean() < minimum:
        #             minimum = param.grad.abs().mean()
        #         elif param.grad.abs().mean() > maximum:
        #             maximum = param.grad.abs().mean()
        #         simple_sum += param.grad.abs().mean()
        #         simple_count += 1
        #     simple_mean = simple_sum / simple_count
        #     print("min:", minimum, "max:", maximum, "mean:", simple_mean)
        # except AttributeError as e:
        #     raise AttributeError("{}. Maybe the parameter '{}' in model is not used, please have a check.".format(e, name))

        # return simple_mean

    # complex implememtation
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()
    m1_next, m2_next = True, True
    while m1_next or m2_next:
        try:
            name, param = params1.__next__()
            display("grad of " + name, param.grad)
        except StopIteration:
            # print("stop1")
            m1_next = False
        try:
            name, param = params2.__next__()
            display("grad of " + name, param.grad)
        except StopIteration:
            # print("stop2")
            m2_next = False


def check_optimizer(optimizer):
    r"""Check state of optimizer for reproduce other's work.
    Usage:
        check_optimizer(optimizer)
    """
    if isinstance(optimizer, torch.optim.SGD):
        keys = ["momentum_buffer"]

    print(optimizer)
    for i, (group, param_group) in enumerate(
        zip(optimizer.param_groups, optimizer.state_dict()["param_groups"])):
        name = group.get("name", "Unnamed {}".format(i))
        print("{}: {} layers of params".format(name, len(group["params"])))
        state = optimizer.state_dict()["state"]
        for num in param_group["params"]:
            for key in keys:
                display("{} of layer {}".format(key, num), state[num][key])


def register_hook(model,
                  trigger="backward",
                  allow_base_modules_only=True,
                  layer_wise=True):
    r"""Register a hook for all base module.
    conv, fc, Sequential module and so on, will be visited, but at default only base module.
    Args:
        model (nn.Module): the model need to be add hook
        hook_fn (nn.Module): the hook function need to be register (deprecated)
        trigger (forward_pre | forward | backward (default)): which moment the hook trigger
        allow_base_modules_only (bool): it control whether create summary for those middle modules
        layer_wise (bool): only for 'forward' trigger, whether do more exact checking
    Example 1:
        hooks = register_hook(model.backbone, "forward", layer_wise=True)
        data = torch.ones(2, 3, 224, 224)
        model.eval()
        output = model.backbone(data) # information will be printed when running this line
        for h in hooks:
            h.remove()
        output = model.backbone(data) # after hook removed, nothing will be printed when running this line
    Example 2:
        hooks = register_hook(model.backbone, "backward")
        ## forward and compute loss
        loss.backward() # gradient information will be printed when backward loss
    """
    # modules in base_classes will be add a hook_fn, while others not
    # add base module name if needed
    base_classes = ["Linear",
                    "Conv2d",
                    "Flatten",
                    "ReLU",
                    "PReLU",
                    "Sigmoid",
                    "Dropout",
                    "BatchNorm1d",
                    "BatchNorm2d",
                    "MaxPool2d",
                    "AdaptiveAvgPool2d"]
    # create properties
    summary = OrderedDict()
    classes_idx = {}
    hooks = [] # hooks is used to record added hook for removing them later
    assert trigger in ["forward_pre", "forward", "backward"], \
        "trigger should be in ['forward_pre', 'forward', 'backward'], but got {}".format(trigger)
    # register_fn = "register_{}_hook".format(trigger)

    def get_sth(output, sth):
        if isinstance(output, tuple): # if the model has more than one output, "output" here will be a tuple
            result = {}
            for i, _ in enumerate(output):
                result[i] = OrderedDict()
                result[i] = get_sth(output[i], sth)
        elif sth == "sum":
            result = output.sum().item()
        elif sth == "max":
            result = output.max().item()
        elif sth == "min":
            result = output.min().item()
        elif sth == "mean":
            result = output.mean().item()
        elif sth == "size":
            result = list(output.size())
        return result

    def forward_hook(module, input, output):
        r"""An example of forward hook_fn, it will be triggered after model.forward(input) operation.
        Used to check model's input and output.
        Args:
            module (nn.Module): a module
            input (torch.Tensor): input of this module
            output (torch.Tensor): output of this module
        Note that forward_hook has to have no return value.
        """
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)
        if allow_base_modules_only:
            if class_name not in base_classes:
                return
        class_idx = classes_idx.get(class_name)
        if class_idx is None:
            class_idx = 0
            classes_idx[class_name] = 1
        else:
            classes_idx[class_name] += 1

        m_key = "{}-{} ({})".format(class_name, class_idx+1, module_idx+1)
        summary[m_key] = OrderedDict()
        summary[m_key]["input_shape"] = list(input[0].size()) # input is a tuple whose first element is a tensor
        summary[m_key]["output_shape"] = get_sth(output, "size")
        if layer_wise: # more exact checking
            summary[m_key]["input_sum"] = get_sth(input[0], "sum")
            summary[m_key]["output_sum"] = get_sth(output, "sum")
            summary[m_key]["output_max"] = get_sth(output, "max")
            summary[m_key]["output_min"] = get_sth(output, "min")
            summary[m_key]["output_mean"] = get_sth(output, "mean")

        params = 0
        if hasattr(module, "weight"):
            params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
            if module.weight.requires_grad:
                summary[m_key]["trainable"] = True
            else:
                summary[m_key]["trainable"] = False
        #if hasattr(module, "bias"):
        #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))

        summary[m_key]["num_params"] = params # not take bias into consideration
        pprint.pprint({m_key: summary[m_key]})

    def backward_hook(module, grad_input, grad_output):
        r"""An example of backward hook_fn, it will be triggered after module.backward() operation.
        Used to check model's gradient.
        Args:
            module (nn.Module): a module
            grad_input (torch.Tensor): gradient of this module's input
            grad_output (torch.Tensor): gradient of this module's output
        """
        info_dict = {"Sigmoid": ["downstream"],
                     "Linear": ["bias", "downstream", "weight"],
                     "ReLU": ["downstream"],
                     "Conv2d": ["downstream", "weight", "bias"],
                     "BatchNorm1d": ["downstream", "weight", "bias"],
                     "BatchNorm2d": ["downstream", "weight", "bias"],
                     "AdaptiveAvgPool2d": ["downstream"],
                     "MaxPool2d": ["downstream"]}
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)
        # if allow_base_modules_only:
        #     if class_name not in base_classes:
        #         return
        class_idx = classes_idx.get(class_name)
        if class_idx is None:
            class_idx = 0
            classes_idx[class_name] = 1
        else:
            classes_idx[class_name] += 1
        # the order of backward is opposite to the one of forward, so here use a negative order
        m_key = "{} -{} (-{})".format(class_name, class_idx+1, module_idx+1)
        print("{}:".format(m_key))
        summary[m_key] = OrderedDict()

        if class_name in info_dict:
            for i, gin in enumerate(grad_input):
                if gin is None: # the beginning network layer has no grad for downstream
                    print("grad to {}: None".format(info_dict[class_name][i]))
                else:
                    display("grad to {}".format(info_dict[class_name][i]), gin)
        else:
            for i, gin in enumerate(grad_input):
                display("grad_input {}".format(i + 1), gin)

        # it seems that all base module have only one output
        display("grad from upstream", grad_output[0])

    if trigger == "forward_pre":
        raise NotImplementedError(trigger)
    elif trigger == "forward":
        hook_fn = forward_hook
    elif trigger == "backward":
        hook_fn = backward_hook

    def _register_hook(module):
        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not module == model: # make sure "module" is a base module, such conv, fc and so on
            if trigger == "forward_pre":
                hooks.append(module.register_forward_pre_hook(hook_fn))
            elif trigger == "forward":
                hooks.append(module.register_forward_hook(hook_fn))
            elif trigger == "backward":
                hooks.append(module.register_backward_hook(hook_fn))

    # recursively register hook on each network module, including base module(Conv, FC),
    # middle module(Sequential) and top module(model)
    model.apply(_register_hook)

    return hooks

