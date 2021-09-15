# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import sys
import ast
import shutil
import tempfile

from importlib import import_module

from .base import BaseFileHandler


def validate_py_syntax(filename: str):
    with open(filename) as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"There are syntax errors in config file {filename}: {e}")


class PythonHandler(BaseFileHandler):
    def load_from_fileobj(self, file):
        raise NotImplementedError(f"can not load python from TextIOWrapper, " f"please load python file by its path")

    def load_from_path(self, filepath):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py")

            temp_config_name = os.path.basename(temp_config_file.name)
            # Substitute predefined variables
            shutil.copyfile(filepath, temp_config_file.name)

            temp_module_name = os.path.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            validate_py_syntax(filepath)
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            obj = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
            # delete imported module
            del sys.modules[temp_module_name]

            # close temp file
            temp_config_file.close()

        return obj

    def dump_to_fileobj(self, obj, file, **kwargs):
        raise NotImplementedError(f"can not dump the obj as python file")

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError(f"can not dump the obj as string in python file")
