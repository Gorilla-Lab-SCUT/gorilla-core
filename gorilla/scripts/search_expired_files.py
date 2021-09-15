# Copyright (c) Gorilla-Lab. All rights reserved.
# some codes for maintaining codebase or system

import os
import argparse


def search_expired_files(user, base_dir, days=60, file_fmts=["*.pth", "*.pt"]):
    r"""Author: zhang.haojian
    Search expired files belonging to someone, which are more likely to be removed.
    Args:
        user (str): files whose owner is `user` will be selected
        base_dir (str): directory of searching
        days (int): files whose modified time are `days` days before now will be selected
        file_fmts (list): file formats that want to search

    Example:
        search_expired_files("lab-zhang.haojian")
    """
    shfile = os.path.join(os.path.dirname(__file__), "search_expired_files.sh")
    os.system(
        f"""sh {shfile} {user} '{base_dir}' {days} '{" ".join(file_fmts)}' """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Search expired files belonging to someone, "
        "which are more likely to be removed. \n\t"
        "It will output the `expired_files_$(date '+%Y-%m-%d').txt` file, "
        "which contains the search file results")
    parser.add_argument("--user",
                        type=str,
                        help="files whose owner is `user` will be selected")
    parser.add_argument("--base-dir",
                        type=str,
                        default="/",
                        help="directory of searching")
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help=
        "files whose modified time are `days` days before now will be selected"
    )
    parser.add_argument(
        "--file-fmts",
        nargs='+',
        default=["*.pth", "*.pt"],
        help="file formats that want to search (support list input)")
    opt = parser.parse_args()

    # run the search
    if opt.base_dir.endswith("/"):
        opt.base_dir = opt.base_dir[:-1]

    search_expired_files(opt.user, opt.base_dir, opt.days, opt.file_fmts)
