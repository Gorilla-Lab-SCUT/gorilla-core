# Copyright (c) Gorilla-Lab. All rights reserved.
# some codes for maintaining codebase or system

import os

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
