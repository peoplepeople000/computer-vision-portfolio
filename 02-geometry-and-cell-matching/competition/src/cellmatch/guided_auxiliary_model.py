"""Paired-cell guide with an all-cell segmentation training head."""
import torch
from torch import nn
from torch.nn import functional as F
from .guided_ex_repaired import GuidedNet, objective as paired_objective
from .learning import loss_function

class AuxiliaryGuidedNet(GuidedNet):
    def __init__(self, width=24):
        super().__init__(width)
        self.auxiliary = nn.Conv2d(width, 2, 1)

    def forward(self, x, auxiliary=False):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        result = self.out(e), self.presence(c.mean((2, 3))).ravel()
        return (*result, self.auxiliary(e)) if auxiliary else result

def objective(logits, presence, auxiliary, target, exists):
    return paired_objective(logits, presence, target[:, :2], exists) + loss_function(auxiliary, target[:, 2:])
