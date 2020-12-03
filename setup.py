import os
import os.path as osp
import sys
from glob import glob
from setuptools import dist, setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
    EXT_TYPE = "pytorch"
except ModuleNotFoundError:
    from Cython.Distutils import build_ext as BuildExtension
    print("Skip building ext ops due to the absence of torch.")


def get_requirements(filename="requirements.txt"):
    assert osp.exists(filename), "{} not exists".format(filename)
    with open(filename, "r") as f:
        content = f.read()
    lines = content.split("\n")
    requirements_list = list(
        filter(lambda x: x != "" and not x.startswith("#"), lines))
    return requirements_list


def get_version():
    version_file = osp.join("gorilla", "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


if __name__ == "__main__":
    setup(
        name="gorilla-core",
        version=get_version(),
        author="Gorilla Authors",
        author_email="mszhihaoliang@mail.scut.edu.cn",
        description="ToolBox Package for Gorilla-Lab using PyTorch",
        long_description=open("README.md").read(),
        license="MIT",
        install_requires=get_requirements(),
        packages=find_packages(exclude=["tests"]),
        zip_safe=False)
