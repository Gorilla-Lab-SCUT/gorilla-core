# Copyright (c) Gorilla-Lab. All rights reserved.

import random
import time

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
            print("{}{}: type={} value={}".format(" "*ind, kw, strip(type(arg)), arg))
        elif isinstance(arg, (list, tuple)):
            other = "..." if len(arg) > dl else ""
            print("{}{}: type={} value={}{} len={}".format(" "*ind, kw, strip(type(arg)), arg[:dl], other, len(arg)))
        elif isinstance(arg, set):
            print("{}{}: type={} len={}".format(" "*ind, kw, strip(type(arg)), len(arg)))
            for i, value in enumerate(arg):
                go("Unnamed {}".format(i+1), value, dl, ind+4)
        elif isinstance(arg, dict):
            other = "..." if len(arg.keys()) > dl else ""
            print("{}{}: type={} key={}{} len={}".format(" "*ind, kw, strip(type(arg)), list(arg.keys())[:dl], other, len(arg)))
            for key, value in arg.items():
                go(key, value, dl, ind+4)
        elif isinstance(arg, np.ndarray):
            print("{}{}: type={} shape={} dtype={}".format(" "*ind, kw, strip(type(arg)), arg.shape, arg.dtype))
        elif isinstance(arg, torch.Tensor):
            print("{}{}: type={} size={} dtype={}".format(" "*ind, kw, strip(type(arg)), arg.size(), arg.dtype))
        elif isinstance(arg, (int, float)):
            print("{}{}: type={} value={}".format(" "*ind, kw, strip(type(arg)), arg))
        else:
            try:
                print("{}{}: type={} len={}".format(" "*ind, kw, strip(type(arg)), len(arg)))
            except TypeError:
                print("{}{}: type={}".format(" "*ind, kw, strip(type(arg))))
            
    for i, arg in enumerate(args):
        go("Unnamed {}".format(i+1), arg)
    for kw, arg in kwargs.items():
        go(kw, arg)

def display(name, param):
    r"""This function can be used to debug in data loading pipeline and model forwarding."""
    if isinstance(param, torch.Tensor):
        print("{} max: {:+.5f} min: {:+.5f} mean: {:+.5f} abs mean: {:+.5f} size:{}".format(
            name.ljust(45), param.max().item(), param.min().item(),
            param.mean().item(), param.abs().mean().item(), list(param.size())))
    elif isinstance(param, np.ndarray):
        print("{} max: {:+.5f} min: {:+.5f} mean: {:+.5f} shape:{}".format(
            name.ljust(15), param.max().item(), param.min().item(),
            param.mean().item(), param.shape))
    elif isinstance(param, str):
        print("{}: {}".format(name, param))
    else:
        raise NotImplementedError("type {}".format(type(param)))


def check_rand_state():
    r"""Check state of random number generator of numpy, random and torch"""
    # only print the first n element for brevity
    n = 10
    state = random.getstate()
    print("random: state: {}, counter: {}".format(state[1][:n], state[1][-1]))
    state = np.random.get_state()
    print("numpy: state: {}, counter: {}".format(list(state[1][:n]), state[2]))
    state = torch.get_rng_state()
    def uint8_to_uint32(u8list):
        r"""Concat a list of 4 uint8 number to an uint32 number.
        validated supported input type: torch.ByteTensor, list
        """
        return (int(u8list[3])<<24) + (int(u8list[2])<<16) + (int(u8list[1])<<8) + int(u8list[0])
    front_state = [uint8_to_uint32(state[24+8*i: 24+8*i+4]) for i in range(n-1)]
    print("torch: seed: {}, state: {} counter: {} {}".format(
        uint8_to_uint32(state[:4]),
        front_state,
        int(state[8]) + (int(state[9])<<8),
        int(state[16]) + (int(state[17])<<8)))

