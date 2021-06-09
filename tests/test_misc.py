# Copyright (c) Open-MMLab. All rights reserved.
import pytest

import gorilla


def test_iter_cast():
    assert gorilla.list_cast([1, 2, 3], int) == [1, 2, 3]
    assert gorilla.list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert gorilla.list_cast([1, 2, 3], str) == ['1', '2', '3']
    assert gorilla.tuple_cast((1, 2, 3), str) == ('1', '2', '3')
    assert next(gorilla.iter_cast([1, 2, 3], str)) == '1'
    with pytest.raises(TypeError):
        gorilla.iter_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        gorilla.iter_cast(1, str)


def test_is_seq_of():
    assert gorilla.is_seq_of([1.0, 2.0, 3.0], float)
    assert gorilla.is_seq_of([(1, ), (2, ), (3, )], tuple)
    assert gorilla.is_seq_of((1.0, 2.0, 3.0), float)
    assert gorilla.is_list_of([1.0, 2.0, 3.0], float)
    assert not gorilla.is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not gorilla.is_tuple_of([1.0, 2.0, 3.0], float)
    assert not gorilla.is_seq_of([1.0, 2, 3], int)
    assert not gorilla.is_seq_of((1.0, 2, 3), int)


def test_slice_list():
    in_list = [1, 2, 3, 4, 5, 6]
    assert gorilla.slice_list(in_list, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert gorilla.slice_list(in_list, [len(in_list)]) == [in_list]
    with pytest.raises(TypeError):
        gorilla.slice_list(in_list, 2.0)
    with pytest.raises(ValueError):
        gorilla.slice_list(in_list, [1, 2])


def test_concat_list():
    assert gorilla.concat_list([[1, 2]]) == [1, 2]
    assert gorilla.concat_list([[1, 2], [3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]


def test_requires_package(capsys):

    @gorilla.requires_package('nnn')
    def func_a():
        pass

    @gorilla.requires_package(['numpy', 'n1', 'n2'])
    def func_b():
        pass

    @gorilla.requires_package('numpy')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ("Prerequisites 'nnn' are required in method 'func_a' but "
                   "not found, please install them first.\n")

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        "Prerequisites 'n1, n2' are required in method 'func_b' "
        "but not found, please install them first.\n")

    assert func_c() == 1


def test_requires_executable(capsys):

    @gorilla.requires_executable('nnn')
    def func_a():
        pass

    @gorilla.requires_executable(['ls', 'n1', 'n2'])
    def func_b():
        pass

    @gorilla.requires_executable('mv')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ("Prerequisites 'nnn' are required in method 'func_a' "
                   "but not found, please install them first.\n")

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        "Prerequisites 'n1, n2' are required in method 'func_b' "
        "but not found, please install them first.\n")

    assert func_c() == 1


def test_import_modules_from_strings():
    # multiple imports
    import os.path as osp_
    import sys as sys_
    osp, sys = gorilla.import_modules_from_strings(['os.path', 'sys'])
    assert osp == osp_
    assert sys == sys_

    # single imports
    osp = gorilla.import_modules_from_strings('os.path')
    assert osp == osp_
    # No imports
    assert gorilla.import_modules_from_strings(None) is None
    assert gorilla.import_modules_from_strings([]) is None
    assert gorilla.import_modules_from_strings('') is None
    # Unsupported types
    with pytest.raises(TypeError):
        gorilla.import_modules_from_strings(1)
    with pytest.raises(TypeError):
        gorilla.import_modules_from_strings([1])
    # Failed imports
    with pytest.raises(ImportError):
        gorilla.import_modules_from_strings('_not_implemented_module')
    with pytest.warns(UserWarning):
        imported = gorilla.import_modules_from_strings(
            '_not_implemented_module', allow_failed_imports=True)
        assert imported is None
    with pytest.warns(UserWarning):
        imported = gorilla.import_modules_from_strings(
            ['os.path', '_not_implemented'], allow_failed_imports=True)
        assert imported[0] == osp
        assert imported[1] is None
