# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import glob
import shutil
import warnings
import logging
from typing import List, Optional

import torch.distributed as dist

from ..core import master_only
from ..version import __version__


@master_only
def backup(backup_dir: str,
           backup_list: [List[str], str],
           contain_suffix: List = ["*.py"],
           strict: bool = False,
           **kwargs) -> None:
    r"""Author: liang.zhihao
    The backup helper function

    Args:
        backup_dir (str): the backup directory
        backup_list (str or List of str): the backup members
        strict (bool, optional): tolerate backup members missing or not.
            Defaults to False.
    """
    logger = logging.getLogger(__name__)

    # if exist, remove the backup dir to avoid copytree exist error
    if osp.exists(backup_dir):
        shutil.rmtree(backup_dir)

    os.makedirs(backup_dir, exist_ok=True)
    # log gorilla version
    logger.info(f"gorilla-core version is {__version__}")
    try:
        from gorilla2d import __version__ as g2_ver
        logger.info(f"gorilla2d version is {g2_ver}")
    except:
        pass
    try:
        from gorilla3d import __version__ as g3_ver
        logger.info(f"gorilla3d version is {g3_ver}")
    except:
        pass

    logger.info(f"backup files at {backup_dir}")
    if not isinstance(backup_list, list):
        backup_list = [backup_list]
    if not isinstance(contain_suffix, list):
        contain_suffix = [contain_suffix]

    for name in backup_list:
        # deal with missing file or dir
        miss_flag = (not osp.exists(name))
        if miss_flag:
            if strict:
                raise FileExistsError(f"{name} not exist")
            warnings.warn(f"{name} not exist")

        # dangerous dir warning
        if name in ["data", "log"]:
            warnings.warn(f"'{name}' maybe the unsuitable to backup")
        if osp.isfile(name):
            # just copy the filename
            dst_name = name.split("/")[-1]
            shutil.copy(name, osp.join(backup_dir, dst_name))
        if osp.isdir(name):
            # only match '.py' files
            files = glob.iglob(osp.join(name, "**", "*.*"), recursive=True)
            ignore_suffix = set(
                map(lambda x: "*." + x.split("/")[-1].split(".")[-1], files))
            for suffix in contain_suffix:
                if suffix in ignore_suffix:
                    ignore_suffix.remove(suffix)
            # copy dir
            shutil.copytree(name,
                            osp.join(backup_dir, name),
                            ignore=shutil.ignore_patterns(*ignore_suffix))
