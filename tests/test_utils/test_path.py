# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import pytest

import gorilla


def test_is_filepath():
    assert gorilla.is_filepath(__file__)
    assert gorilla.is_filepath('abc')
    assert gorilla.is_filepath(Path('/etc'))
    assert not gorilla.is_filepath(0)


def test_fopen():
    assert hasattr(gorilla.fopen(__file__), 'read')
    assert hasattr(gorilla.fopen(Path(__file__)), 'read')


def test_check_file_exist():
    gorilla.check_file(__file__)
    with pytest.raises(FileNotFoundError):
        gorilla.check_file('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(osp.dirname(__file__)), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']
    assert set(gorilla.scandir(folder)) == set(filenames)
    assert set(gorilla.scandir(Path(folder))) == set(filenames)
    assert set(gorilla.scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(gorilla.scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(gorilla.scandir(folder, '.png')) == set()

    filenames_recursive = [
        'a.bin', '1.txt', '2.txt', '1.json', '2.json', 'sub/1.json',
        'sub/1.txt'
    ]
    assert set(gorilla.scandir(folder,
                            recursive=True)) == set(filenames_recursive)
    assert set(gorilla.scandir(Path(folder),
                            recursive=True)) == set(filenames_recursive)
    assert set(gorilla.scandir(folder, '.txt', recursive=True)) == set([
        filename for filename in filenames_recursive
        if filename.endswith('.txt')
    ])
    with pytest.raises(TypeError):
        list(gorilla.scandir(123))
    with pytest.raises(TypeError):
        list(gorilla.scandir(folder, 111))
