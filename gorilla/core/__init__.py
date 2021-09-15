# Copyright (c) Gorilla-Lab. All rights reserved.
from .misc import (concat_list, convert_list, convert_list_str,
                   convert_list_int, convert_list_float, iter_cast, slice_list,
                   iter_cast, list_cast, tuple_cast, check_prerequisites,
                   deprecated_api_warning, is_seq_of, is_list_of, is_tuple_of,
                   multi_apply, is_multiple, is_power2, requires_package,
                   requires_executable, check_prerequisites,
                   import_modules_from_strings)

from .comm import (get_world_size, get_rank, get_local_rank, get_local_size,
                   is_main_process, synchronize, gather, all_gather,
                   shared_random_seed, reduce_dict, get_dist_info, master_only)

from .env import set_random_seed, collect_env_info

from .fp16 import cast_tensor_type, auto_fp16, force_fp32

from .launch import launch

from .setup import default_argument_parser

from .registry import Registry, build_from_cfg, auto_registry, obj_from_dict

HOOKS = Registry("hooks")
LOSSES = Registry("losses")
MODELS = Registry("models")
MODULES = Registry("modules")
DATASETS = Registry("datasets")
PIPELINES = Registry("pipelines")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")

from functools import partial
build_loss = partial(build_from_cfg, registry=LOSSES)
build_hook = partial(build_from_cfg, registry=HOOKS)
build_model = partial(build_from_cfg, registry=MODELS)
build_module = partial(build_from_cfg, registry=MODULES)
build_pipeline = partial(build_from_cfg, registry=PIPELINES)

# the inner func for build_optimizer in solver.build
_build_optimizer = partial(build_from_cfg, registry=OPTIMIZERS)
_build_scheduler = partial(build_from_cfg, registry=SCHEDULERS)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
