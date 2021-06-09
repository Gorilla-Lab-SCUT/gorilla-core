# Copyright (c) Open-MMLab. All rights reserved.
import time

import pytest

import gorilla


def test_timer_init():
    timer = gorilla.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = gorilla.Timer()
    assert timer.is_running


def test_timer_run():
    timer = gorilla.Timer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1e-2
    time.sleep(1)
    assert abs(timer.since_last() - 1) < 1e-2
    assert abs(timer.since_start() - 2) < 1e-2
    timer = gorilla.Timer(start=False)
    with pytest.raises(gorilla.TimerError):
        timer.since_start()
    with pytest.raises(gorilla.TimerError):
        timer.since_last()


def test_timer_context(capsys):
    with gorilla.Timer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1e-2
    with gorilla.Timer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n'
