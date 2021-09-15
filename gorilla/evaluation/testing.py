# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import pprint
import sys
from collections import OrderedDict
from collections.abc import Mapping


def print_csv_format(results):
    r"""
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(
        results,
        OrderedDict), results  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        # Don't print "AP-category" metrics since they are usually not tracked.
        important_res = [(k, v) for k, v in res.items() if "-" not in k]
        logger.info(f"copypaste: Task: {task}")
        logger.info(f"copypaste: " + ",".join([k[0] for k in important_res]))
        logger.info(f"copypaste: " +
                    ",".join([f"{k[1]:.4f}" for k in important_res]))


def verify_results(cfg, results):
    r"""
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task].get(metric, None)
        if actual is None:
            ok = False
            continue
        if not np.isfinite(actual):
            ok = False
            continue
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    r"""
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
