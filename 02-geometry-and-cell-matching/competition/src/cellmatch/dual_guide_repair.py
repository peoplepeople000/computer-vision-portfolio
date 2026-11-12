"""Small image/native/public-mask contour repair with explicit empty abstention."""
import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1),
                         nn.GroupNorm(4, outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(4, outputs), nn.ReLU())


class DualGuideRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = Block(3, 16), Block(16, 32), Block(32, 64)
        self.dec2, self.dec1 = Block(96, 32), Block(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        gray = x[:, :1]
        mean = gray.mean((2, 3), keepdim=True)
        sd = (gray - mean).square().mean((2, 3), keepdim=True).sqrt().clamp_min(.02)
        gray = ((gray - mean) / sd).clamp(-3, 3) / 3
        a = self.enc1(torch.cat([gray, x[:, 1:]], 1))
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return self.output(e)


def loss_fn(logits, target):
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


def decode(probability, prior):
    """Native-grid threshold/component selection. Empty never restores prior."""
    resized = F.interpolate(torch.as_tensor(probability)[None, None].float(),
                            size=prior.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
    labels, count = ndimage.label(resized >= .5)  # Fixed four-connectivity.
    overlap = np.bincount(labels[prior].ravel(), minlength=count + 1)
    overlap[0] = 0
    result = np.zeros(prior.shape, bool)
    if overlap.max(initial=0) > 0:
        result = labels == int(overlap.argmax())
        if result.sum() < 12:
            result[:] = False
    return result
