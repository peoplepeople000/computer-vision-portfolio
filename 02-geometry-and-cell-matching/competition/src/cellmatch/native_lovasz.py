"""Batched binary per-cell Lovasz hinge, following the MIT author reference.

Reference: Q5/vendor/lovasz-reference/lovasz_losses.py and accompanying LICENSE.
Vectorization preserves the per-image foreground Jaccard formula; this does not
directly optimize strict instance PQ or paired correspondence F1.
"""
import torch
from torch.nn import functional as F
from .learning import loss_function


def lovasz_hinge(logits, labels):
    assert logits.shape == labels.shape and logits.ndim == 3
    scores = logits.reshape(logits.shape[0], -1)
    target = labels.reshape(labels.shape[0], -1).float()
    errors = 1 - scores * (2 * target - 1)
    ordered_errors, order = torch.sort(errors, dim=1, descending=True)
    ordered_target = torch.gather(target, 1, order)
    positives = ordered_target.sum(1, keepdim=True)
    intersection = positives - ordered_target.cumsum(1)
    union = positives + (1 - ordered_target).cumsum(1)
    jaccard = 1 - intersection / union
    gradient = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], 1)
    return (F.relu(ordered_errors) * gradient).sum(1).mean()


def loss(logits, targets, step):
    if step <= 1000:
        return loss_function(logits, targets)
    fg, center = targets[:, :1], targets[:, 1:]
    bce = F.binary_cross_entropy_with_logits(logits[:, :1], fg, pos_weight=torch.tensor(2., device=logits.device))
    distance = (F.smooth_l1_loss(logits[:, 1:].sigmoid(), center, reduction='none') * (1 + 4 * fg)).mean()
    return bce + lovasz_hinge(logits[:, 0], targets[:, 0]) + 2 * distance
