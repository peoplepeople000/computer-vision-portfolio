"""Training-label support matched to the raw branch nine-pixel receptive field."""
import torch
from torch.nn import functional as F


def supervision_support(target):
    assert target.ndim==4 and target.shape[1]==1
    return F.max_pool2d(target,9,stride=1,padding=4)


def boundary_supervised_loss(logits,target,valid=None):
    valid=supervision_support(target) if valid is None else valid
    probability=logits.sigmoid();den=valid.sum().clamp_min(1.)
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*valid).sum()/den
    ratio=(2*(probability*target*valid).sum()+1)/(((probability+target)*valid).sum()+1)
    return (bce+1)-ratio
