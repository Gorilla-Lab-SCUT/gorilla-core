# Copyright (c) Gorilla-Lab. All rights reserved.

import torch
import torch.nn.functional as F

def iou_guided_loss(
        scores: torch.Tensor,
        gt_ious: torch.Tensor,
        fg_thresh: float=1.0,
        bg_thresh: float=0.0,
        reduction: str="none",
        use_sigmoid: bool=True,
) -> torch.Tensor:
    r"""
    IoU-guided NMS Loss
    https://arxiv.org/abs/1807.11590
    Args:
        gt_iou (Tensor): ground truth of bbox or mask.
        fg_thresh (float): fore ground threshold.
        bg_thresh (float): back ground threshold.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    """
    # split the gt_iou into three parts
    fg_mask = gt_ious > fg_thresh
    bg_mask = gt_ious < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    gt_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    gt_scores[interval_mask] = gt_ious[interval_mask] * k + b
    
    if use_sigmoid:
        scores = torch.sigmoid(scores)
    loss = F.binary_cross_entropy(scores, gt_scores, reduction=reduction)

    return loss


