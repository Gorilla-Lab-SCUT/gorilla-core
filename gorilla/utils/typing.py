# Copyright (c) Gorilla-Lab. All rights reserved.
import functools

import torch
import numpy as np


def type_assert(arg):
    r"""Assert type in [`list`, `tuple`, `np.array`, `torch.Tensor`]

    Args:
        arg (instance): instance to be asserted its type
    """
    type_flag = isinstance(arg, list) or \
                isinstance(arg, tuple) or \
                isinstance(arg, np.ndarray) or \
                isinstance(arg, torch.Tensor)
    assert type_flag, ("args type {} not in [`list`, `tuple`, "
                       "`np.ndarray`, `torch.Tensor`]".format(type(arg[0])))


def convert_into_torch_tensor(array) -> torch.Tensor:
    r"""Convert other type array into torch.Tensor

    Args:
        array (list | tuple | obj:`ndarray` | obj:`Tensor`): Input array

    Returns:
        torch.Tensor: Processed array
    """
    type_assert(array)
    if not isinstance(array, torch.Tensor):
        array = torch.Tensor(array)
    array = array.squeeze().float()
    return array


def convert_into_nparray(array, dtype=np.float32) -> np.array:
    r"""Convert other type array into np.array

    Args:
        array (list | tuple | obj:`ndarray` | obj:`Tensor`): Input array

    Returns:
        torch.Tensor: Processed array
    """
    type_assert(array)
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = array.squeeze().astype(dtype)
    return array
    

def assert_and_auto_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        type_assert(args[0])
        be_numpy = True if not args[0].__class__==torch.Tensor else False
        if be_numpy:
            return func(*args, **kwargs).cpu().numpy()
        else:
            return func(*args, **kwargs)

    return wrapper


def to_float32(arr: np.ndarray) -> np.ndarray:
    r"""Author: lei.jiabao
    process float16 array specially
    
    Args:
        arr (np.ndarray): the origin array
    
    Returns:
        np.ndarray: array as float32
    """
    if arr.dtype == np.float16:
        return arr.astype(np.float32) + 1e-4 * np.random.randn(*arr.shape).astype(np.float32)
    elif arr.dtype == np.float32:
        return arr
    else:
        return arr.astype(np.float32)


