import torch
import torch.nn as nn


class LabelSmoothCELoss(nn.Module):
    r"""
    Cross-entrophy loss with label smooth.

    Args:
        epsilon: Smoothing level. Use one-hot label when set to 0, use uniform label when set to 1.
    """
    def __init__(self, epsilon):
        super(LabelSmoothCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: A float tensor of shape: (minibatch, C).
            targets: A float tensor of shape: (minibatch,). Stores the class indices
                    in range `[0, C - 1]`.
        Returns:
            A scalar tensor.
        """
        loss = label_smooth_ce_loss(logits, targets, self.epsilon)
        return loss


def label_smooth_ce_loss(logits: torch.Tensor, targets: torch.Tensor,
                         epsilon: float) -> torch.Tensor:
    r"""
    Cross-entrophy loss with label smooth.

    Args:
        logits: A float tensor of shape: (minibatch, C).
        targets: A float tensor of shape: (minibatch,). Stores the class indices
                 in range `[0, C - 1]`.
        epsilon: Smoothing level. Use one-hot label when set to 0, use uniform label when set to 1.

    Returns:
        A scalar tensor.
    """
    log_probs = nn.functional.log_softmax(logits, dim=1)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + epsilon / logits.shape[1]
    loss = (-targets * log_probs).mean(0).sum()
    return loss
