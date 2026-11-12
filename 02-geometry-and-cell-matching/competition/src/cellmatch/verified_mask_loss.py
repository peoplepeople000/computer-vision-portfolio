"""Suppress segmentation gradients on unverified prompts, retain credit supervision."""
import torch
from torch.nn import functional as F
from cellmatch.learning import loss_function

def verified_mask_loss(logits, presence_logits, targets, presence_targets):
    verified = presence_targets.reshape(-1) > .5
    if verified.any():
        mask_loss = loss_function(logits[verified], targets[verified]) * verified.float().mean()
    else:
        mask_loss = logits.sum() * 0.
    credit_loss = F.binary_cross_entropy_with_logits(presence_logits, presence_targets)
    return mask_loss + .5 * credit_loss
