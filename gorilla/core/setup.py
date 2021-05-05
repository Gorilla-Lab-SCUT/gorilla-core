# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# modify from detectron2 https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
# need to fix

import os
import argparse

def default_argument_parser():
    """
    Create a parser with some common arguments used by gorilla users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="gorilla Training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank",
                        type=int,
                        default=0,
                        help="the rank of this machine (unique per machine)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid()) % 2**14
    parser.add_argument("--dist-url",
                        default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser
