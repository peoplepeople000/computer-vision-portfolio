"""Process a fixed physical32px crop on64px; return native-grid logits.

Interpolation does not recover new optical detail. This changes the U-Net's
internal sampling/receptive field while keeping original32px supervision.
"""
import torch
from torch.nn import functional as F
from .learning import SmallUNet


def double_grid_input(x):
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('Expected N3HW raw/local/binary-prior inputs')
    raw = F.interpolate(x[:, :2], scale_factor=2, mode='bilinear', align_corners=False)
    prior = x[:, 2:3].repeat_interleave(2, -2).repeat_interleave(2, -1)
    return torch.cat((raw, prior), 1)


def native_grid_logits(z):
    return F.avg_pool2d(z, 2)


class DoubleGridRefiner(SmallUNet):
    def forward(self, x):
        return native_grid_logits(super().forward(double_grid_input(x)))
