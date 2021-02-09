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
    assert type_flag, (f"args type {type(arg[0])} not in "
                       f"[`list`, `tuple`, `np.ndarray`, `torch.Tensor`]")


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


def _replace(arg, type="numpy"):
    assert type in ["numpy", "torch"]
    # np.ndarray -> torch.Tensor
    if isinstance(arg, np.ndarray) and type == "torch":
        arg = torch.from_numpy(arg)
    # torch.Tensor -> np.ndarray
    elif isinstance(arg, torch.Tensor) and type == "numpy":
        if arg.requires_grad:
            arg = arg.detach()
        arg = arg.cpu().numpy()
    # keep origin type
    else:
        pass
    return arg


def auto_type(type):
    r"""Author: Liang.Zhihao
    automatically convert the 'np.ndarray' and 'torch.Tensor' according to type

    Args:
        type (str): 'numpy' or 'torch'
    
    Example:
        >>> # numpy auto convert
        >>> @gorilla.auto_type("numpy")
        >>> def test(a, b, c):
        >>>     print(f"a: {type(a)}, b: {type(b)}, c: {type(c)}")
        >>> test(torch.randn(3), np.ones(3), [1, 1, 1])
        a: <class 'numpy.ndarray'>, b: <class 'numpy.ndarray'>, c: <class 'list'>
        >>> # torch auto convert
        >>> @gorilla.auto_type("torch")
        >>> def test(a, b, c):
        >>>     print(f"a: {type(a)}, b: {type(b)}, c: {type(c)}")
        >>> test(torch.randn(3), np.ones(3), [1, 1, 1])
        a: <class 'torch.Tensor'>, b: <class 'torch.Tensor'>, c: <class 'list'>
        >>> # specify arguments
        >>> test(torch.randn(3), c=np.ones(3), b=[1, 1, 1])
        a: <class 'torch.Tensor'>, b: <class 'list'>, c: <class 'torch.Tensor'>
    """
    assert type in ["numpy", "torch"], f"must be 'numpy' or 'torch', but got {type}"
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            replace_args = []
            replace_kwargs = {}
            for arg in args:
                replace_args.append(_replace(arg, type))
            for key, arg in kwargs.items():
                replace_kwargs[key] = _replace(arg, type)
            return func(*replace_args, **replace_kwargs)
        return wrapper
    return actual_decorator


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


