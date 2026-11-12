"""Separate candidate presence and native-identity residual contour outputs."""
import torch
from torch import nn
from torch.nn import functional as F
from .dual_guide_repair import Block, loss_fn


class DualGuideObjectRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = Block(3, 16), Block(16, 32), Block(32, 64)
        self.dec2, self.dec1 = Block(96, 32), Block(48, 16)
        self.contour_delta = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.contour_delta.weight)
        nn.init.zeros_(self.contour_delta.bias)
        self.presence = nn.Linear(128, 1)

    def forward(self, x):
        gray = x[:, :1]
        mean = gray.mean((2, 3), keepdim=True)
        sd = (gray - mean).square().mean((2, 3), keepdim=True).sqrt().clamp_min(.02)
        gray = ((gray - mean) / sd).clamp(-3, 3) / 3
        a = self.enc1(torch.cat([gray, x[:, 1:]], 1))
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        pooled = torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)
        presence = self.presence(pooled)[:, 0]
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        # logit(.9) inside prior, logit(.1) outside; initial residual is zero.
        contour = (2*x[:, 1:2]-1)*2.1972245773362196 + self.contour_delta(e)
        return contour, presence


def object_loss(outputs, target):
    contour, presence = outputs
    positive = target.flatten(1).any(1)
    classification = F.binary_cross_entropy_with_logits(presence, positive.float())
    segmentation = loss_fn(contour[positive], target[positive]) if positive.any() else contour.sum()*0
    return classification + segmentation
