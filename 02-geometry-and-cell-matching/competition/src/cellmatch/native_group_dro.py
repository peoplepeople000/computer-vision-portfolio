"""Per-instance original contour loss and detached group-DRO weighting.

Exponentiated-gradient update follows Sagawa et al., ICLR2020, group_DRO/loss.py.
Local implementation; no generalization adjustment or loss normalization.
"""
import torch
from torch.nn import functional as F


def instance_losses(logits, targets):
    fg, center = targets[:, :1], targets[:, 1:]
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(
        logits[:, :1], fg, pos_weight=logits.new_tensor(2.), reduction='none'
    ).mean(dims)
    dice = 1-(2*(probability[:, :1]*fg).sum(dims)+1)/(
        probability[:, :1].sum(dims)+fg.sum(dims)+1)
    distance = (F.smooth_l1_loss(probability[:, 1:], center, reduction='none')
                *(1+4*fg)).mean(dims)
    return bce+dice+2*distance


def group_means(losses, domains, n_groups=2):
    membership = domains[None] == torch.arange(n_groups, device=domains.device)[:, None]
    count = membership.sum(1)
    mean = (membership.to(losses.dtype)@losses)/count.clamp_min(1)
    return mean, count


class GroupDRO:
    def __init__(self, device, n_groups=2, step_size=.01):
        self.log_weights = torch.zeros(n_groups, device=device)
        self.step_size = step_size

    def __call__(self, losses, domains):
        means, counts = group_means(losses, domains, len(self.log_weights))
        with torch.no_grad():
            self.log_weights.add_(self.step_size*means.detach())
            self.log_weights.sub_(self.log_weights.max())
            weights = self.log_weights.softmax(0)
        return means@weights, means, counts, weights
