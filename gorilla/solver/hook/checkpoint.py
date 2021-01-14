# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp

from .hook import Hook
from ...core import master_only, HOOKS


@HOOKS.register_module()
class CheckpointHook(Hook):
    r"""Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``solver.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs

    @master_only
    def after_epoch(self, solver):
        r"""save checkpoint after each epoch"""
        solver.logger.info(f"Saving checkpoint at {solver.epoch + 1} epochs")
        if not self.out_dir:
            self.out_dir = solver.work_dir
        solver.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get("filename_tmpl", "epoch_{}.pth")
            current_epoch = solver.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = osp.join(self.out_dir, filename_tmpl.format(epoch))
                if osp.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    @master_only
    def after_step(self, solver):
        r"""save checkpoint after each iter (default not use)"""
        if self.by_epoch or not self.every_n_iters(solver, self.interval):
            return

        solver.logger.info(f"Saving checkpoint at {solver.iter + 1} iterations")
        if not self.out_dir:
            self.out_dir = solver.work_dir
        solver.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get("filename_tmpl", "iter_{}.pth")
            current_iter = solver.iter + 1
            for _iter in range(
                    current_iter - self.max_keep_ckpts * self.interval, 0,
                    -self.interval):
                ckpt_path = osp.join(self.out_dir, filename_tmpl.format(_iter))
                if osp.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
