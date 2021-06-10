# Copyright (c) Gorilla-Lab. All rights reserved.
# some codes for maintaining codebase or system

import os
import argparse
from typing import List

def search_expired_files(username, days=60, file_fmts=["*.pth", "*.pt"]):
    r"""Author: zhang.haojian
    Search expired files belonging to someone, which are more likely to be removed.
    Args:
        username (str): user name
        days (int): files whose modified time are `days` days before now will be selected
        file_fmts (list): file formats that want to search

    Example:
        search_expired_files("lab-zhang.haojian")
    """
    shfile = os.path.join(os.path.dirname(__file__), "search_expired_files.sh")
    os.system(f"""sh {shfile} {username} {days} '{" ".join(file_fmts)}' """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search expired files belonging to someone, "
                                     "which are more likely to be removed. \n\t"
                                     "It will output the `expired_files_$(date '+%Y-%m-%d').txt` file, "
                                     "which contains the search file results")
    parser.add_argument("--username",
                        type=str,
                        help="the directory of username to search")
    parser.add_argument("--days",
                        type=int,
                        default=60,
                        help="files whose modified time are `days` days before now will be selected")
    parser.add_argument("--file-fmts",
                        nargs='+',
                        default=["*.pth", "*.pt"],
                        help="file formats that want to search (support list input)")
    opt = parser.parse_args()

    # run the search
    search_expired_files(opt.username,
                         opt.days,
                         opt.file_fmts)


