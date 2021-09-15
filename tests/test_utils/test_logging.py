# Copyright (c) Open-MMLab. All rights reserved.
import re
import tempfile
import logging
from unittest.mock import patch

import pytest

from gorilla import get_logger, print_log

try:
    # piror rich handler
    from rich.logging import RichHandler as StreamHandler
except:
    from logging import StreamHandler


@patch('torch.distributed.get_rank', lambda: 0)
@patch('torch.distributed.is_initialized', lambda: True)
@patch('torch.distributed.is_available', lambda: True)
def test_get_logger_rank0():
    logger = get_logger(name='rank0.pkg1')
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    logger = get_logger(name='rank0.pkg2', log_level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.DEBUG

    with tempfile.NamedTemporaryFile() as f:
        logger = get_logger(name='rank0.pkg3', log_file=f.name)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], StreamHandler)
    assert isinstance(logger.handlers[1], logging.FileHandler)

    logger_pkg3 = get_logger(name='rank0.pkg3')
    assert id(logger_pkg3) == id(logger)

    logger_pkg3 = get_logger(name='rank0.pkg3.subpkg')
    assert logger_pkg3.handlers == logger_pkg3.handlers


@patch('torch.distributed.get_rank', lambda: 1)
@patch('torch.distributed.is_initialized', lambda: True)
@patch('torch.distributed.is_available', lambda: True)
def test_get_logger_rank1():
    logger = get_logger(name='rank1.pkg1')
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    with tempfile.NamedTemporaryFile() as f:
        logger = get_logger(name='rank1.pkg2', log_file=f.name)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.INFO


def test_print_log_print(capsys):
    print_log('welcome', logger=None)
    out, _ = capsys.readouterr()
    assert out == 'welcome\n'


def test_print_log_silent(capsys, caplog):
    print_log('welcome', logger='silent')
    out, _ = capsys.readouterr()
    assert out == ''
    assert len(caplog.records) == 0


def test_print_log_logger(caplog):
    print_log('welcome', logger='gorilla')
    print_log('welcome', logger='gorilla', level=logging.ERROR)


def test_print_log_exception():
    with pytest.raises(TypeError):
        print_log('welcome', logger=0)
