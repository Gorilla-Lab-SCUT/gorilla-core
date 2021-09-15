# Copyright (c) Open-MMLab. All rights reserved.
import sys

import pytest

import gorilla


def test_collect_env():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip('skipping tests that require PyTorch')

    env_info_str = gorilla.collect_env_info()
    expected_keys = [
        'sys.platform',
        'Python',
        'numpy',
        'gorilla',
        'PyTorch',
        'GPU available',
        'torchvision',
        'OpenCV',
    ]
    for key in expected_keys:
        assert key in env_info_str

    if sys.platform != 'win32':
        assert 'GCC' in env_info_str

    env_info_lines = env_info_str.split("\n")
    platform = env_info_lines[1].split(" ")[-1]
    python_version = env_info_lines[2].split("   ")[-1]
    gorilla_version = env_info_lines[4].split(" ")[-2]
    assert platform == sys.platform
    assert python_version == sys.version.replace('\n', '')
    assert gorilla_version == gorilla.__version__
