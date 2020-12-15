# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import glob
import shutil
import warnings
import logging
from typing import List

from ..version import __version__

def backup(log_dir: str,
           backup_list: [List[str], str],
           logger: logging.Logger=None,
           contain_suffix :List=["*.py"], 
           strict: bool=False) -> None:
    r"""Author: liang.zhihao
    The backup helper function

    Args:
        log_dir (str): the bakcup directory({log_dir}/backup)
        backup_list (str or List of str): the backup members
        logger (logging.Logger, optional): logger. Defaults to None.
        strict (bool, optional): tolerate backup members missing or not.
            Defaults to False.
    """

    backup_dir = osp.join(log_dir, "backup")
    # if exist, remove the backup dir to avoid copytree exist error
    if osp.exists(backup_dir):
        shutil.rmtree(backup_dir)

    os.makedirs(backup_dir, exist_ok=True)
    # log gorilla version
    logger.info("gorilla-core version is {}".format(__version__))
    try:
        from gorilla2d import __version__ as g2_ver
        logger.info("gorilla2d version is {}".format(g2_ver))
    except:
        pass
    try:
        from gorilla3d import __version__ as g3_ver
        logger.info("gorilla3d version is {}".format(g3_ver))
    except:
        pass

    logger.info("backup files at {}".format(backup_dir))
    if not isinstance(backup_list, list):
        backup_list = [backup_list]
    if not isinstance(contain_suffix, list):
        contain_suffix = [contain_suffix]

    for name in backup_list:
        # deal with missing file or dir
        miss_flag = (not osp.exists(name))
        if miss_flag:
            if strict:
                raise FileExistsError("{} not exist".format(name))
            warnings.warn("{} not exist".format(name))

        # dangerous dir warning
        if name in ["data", "log"]:
            warnings.warn("{} maybe the unsuitable to backup".format(name))
        if osp.isfile(name):
            shutil.copy(name, osp.join(backup_dir, name))
        if osp.isdir(name):
            # only match '.py' files
            files = glob.iglob(osp.join(name, "**", "*.*"), recursive=True)
            ignore_suffix = set(map(lambda x: "*." + x.split("/")[-1].split(".")[-1], files))
            for suffix in contain_suffix:
                ignore_suffix.remove(suffix)
            # copy dir
            shutil.copytree(name, osp.join(backup_dir, name), ignore=shutil.ignore_patterns(*ignore_suffix))

