import torch
import torch.nn.functional as F
from torch.autograd import Variable

# TODO: complete hinge loss


def lovasz_loss(probas,
                labels,
                with_softmax=True,
                classes="present",
                ignore=None):
    r"""
    Multi-class Lovasz-Softmax loss
        NOTE: the first dimension must be B
              and the second dimension must be C(num_class)
        probas: [B, C, ...] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, ...] Tensor, ground truth labels (between 0 and C - 1)
        classes: "all" for all, "present" for classes present in labels, or a list of classes to average.
        per_sample: compute the loss per sample instead of per batch
        ignore: void class labels
    """
    if with_softmax:
        probas = F.softmax(probas, dim=1)  # [B, C, ...]
    loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
                               classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes="present"):
    r"""
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: "all" for all, "present" for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    # calculate loss for each class
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == "present" and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(
            torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    # average
    losses = sum(losses) / len(losses)
    return losses


def flatten_probas(probas, labels, ignore=None, sigmoid=False):
    r"""
    Flattens predictions in the batch more generally
    """
    if sigmoid:
        # assumes output of a sigmoid layer
        probas = probas.unsqueeze(1)  # [B, 1, ...]
    C = probas.shape[1]
    ndim = len(probas.shape)
    probas = probas.permute(0, *range(2, ndim),
                            1).contiguous().view(-1, C)  # [B*..., C]
    labels = labels.view(-1)  # [B*...]
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]  # [num_valid, C]
    vlabels = labels[valid]  # [num_valid, C]
    return vprobas, vlabels


def lovasz_grad(gt_sorted):
    r"""
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
