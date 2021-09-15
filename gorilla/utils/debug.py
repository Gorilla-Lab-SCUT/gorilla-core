# Copyright (c) Gorilla-Lab. All rights reserved.

import random
import functools

import numpy as np
import torch


def check(*args, **kwargs):
    r"""
    Usage:
        s = "string"
        l = [1, 2, 3, 4, 5, 6]
        ll = ["a", "b", "c"]
        n = np.ones((3, 3))
        t = torch.ones(3, 3)
        d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
        dd = {"a": {"aa": {"aaa": 123, "bbb": 234}, "bb": 23}, "b": 1}
        check(s, l, ll, l=l, ll=ll, n=n, t=t, d=d, dd=dd)    
    """
    def strip(s):
        r"""strip and get the content between two apostrophe
        dtype('float64') --> float64
        <class 'numpy.ndarray'> --> numpy.ndarray
        """
        return str(s).split("'")[1]

    def go(kw, arg, dl=5, ind=0):
        r"""
        Args:
            kw (str): Name of a variable or keyword of a dict
            arg: Variable of any type
            dl (int, optional): Maximum number of elements displayed of a Sequence or dict
            ind (int, optional): Indent of printed text
        """
        if isinstance(arg, str):
            print(f"{' '*ind}{kw}: type={strip(type(arg))} value={arg}")
        elif isinstance(arg, (list, tuple)):
            other = "..." if len(arg) > dl else ""
            print(
                f"{' '*ind}{kw}: type={strip(type(arg))} value={arg[:dl]}{other} len={len(arg)}"
            )
        elif isinstance(arg, set):
            print(f"{' '*ind}{kw}: type={strip(type(arg))} len={len(arg)}")
            for i, value in enumerate(arg):
                go(f"Unnamed {i+1}", value, dl, ind + 4)
        elif isinstance(arg, dict):
            other = "..." if len(arg.keys()) > dl else ""
            print(
                f"{' '*ind}{kw}: type={strip(type(arg))} key={list(arg.keys())[:dl]}{other} len={len(arg)}"
            )
            for key, value in arg.items():
                go(key, value, dl, ind + 4)
        elif isinstance(arg, np.ndarray):
            print(
                f"{' '*ind}{kw}: type={strip(type(arg))} shape={arg.shape} dtype={arg.dtype}"
            )
        elif isinstance(arg, torch.Tensor):
            print(
                f"{' '*ind}{kw}: type={strip(type(arg))} size={arg.size()} dtype={arg.dtype}"
            )
        elif isinstance(arg, (int, float)):
            print(f"{' '*ind}{kw}: type={strip(type(arg))} value={arg}")
        else:
            try:
                print(f"{' '*ind}{kw}: type={strip(type(arg))} len={len(arg)}")
            except TypeError:
                print(f"{' '*ind}{kw}: type={strip(type(arg))}")

    for i, arg in enumerate(args):
        go(f"Unnamed {i+1}", arg)
    for kw, arg in kwargs.items():
        go(kw, arg)


def display(name, param, logger=None):
    r"""This function can be used to debug in data loading pipeline and model forwarding."""
    if logger:
        print = logger.info
    if isinstance(param, torch.Tensor):
        print(f"{name.ljust(45)} "
              f"max: {param.max().item():+.5f} "
              f"min: {param.min().item():+.5f} "
              f"mean: {param.mean().item():+.5f} "
              f"abs mean: {param.abs().mean().item():+.5f} "
              f"size:{list(param.size())}")
    elif isinstance(param, np.ndarray):
        print(f"{name.ljust(15)} "
              f"max: {param.max().item():+.5f} ",
              f"min: {param.min().item():+.5f} ",
              f"mean: {param.mean().item():+.5f} ", f"shape: {param.shape}")
    elif isinstance(param, str):
        print(f"{name}: {param}")
    else:
        raise NotImplementedError(f"type {type(param)}")


def check_rand_state():
    r"""Check state of random number generator of numpy, random and torch"""
    # only print the first n element for brevity
    n = 10
    state = random.getstate()
    print(f"random: state: {state[1][:n]}, counter: {state[1][-1]}")
    state = np.random.get_state()
    print(f"numpy: state: {list(state[1][:n])}, counter: {state[2]}")
    state = torch.get_rng_state()

    def uint8_to_uint32(u8list):
        r"""Concat a list of 4 uint8 number to an uint32 number.
        validated supported input type: torch.ByteTensor, list
        """
        return (int(u8list[3]) << 24) + (int(u8list[2]) << 16) + (
            int(u8list[1]) << 8) + int(u8list[0])

    front_state = [
        uint8_to_uint32(state[24 + 8 * i:24 + 8 * i + 4]) for i in range(n - 1)
    ]
    print(
        f"torch: seed: {uint8_to_uint32(state[:4])}, "
        f"state: {front_state} "
        f"counter: {int(state[8]) + (int(state[9])<<8)} {int(state[16]) + (int(state[17])<<8)}"
    )


def debugtor():
    """Author: liang.zhihao
    decorator of try and except, excuting the set_trace while error
    
    Example:
        >>> @gorilla.debugtor()
        >>> def a(b):
        >>>     a = 1
        >>>     c = a + b # the given a str type `b` will cause bug
        >>>     return c
        >>> c = a(1) # ok
        >>> c = a("1")
            ...
                        from ipdb import set_trace; set_trace()
        -->             return func(*args, **kwargs) # `s` to step in
                    return wrapper
        ipdb> s
        --> @gorilla.debugtor()
            def a(b):
            ...
    """
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                from ipdb import set_trace
                set_trace()
                return func(*args, **kwargs)  # `s` to step in

        return wrapper

    return actual_decorator
