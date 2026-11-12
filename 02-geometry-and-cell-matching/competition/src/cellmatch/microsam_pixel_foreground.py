"""Small raw-pixel foreground residual over frozen microscopy features."""
import torch
from torch import nn
from torch.nn import functional as F


class PixelForeground(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.GELU(),nn.Conv2d(8,8,3,padding=1),nn.GELU())
        self.semantic=nn.Sequential(nn.Conv2d(64,8,1),nn.GELU())
        self.merge=nn.Sequential(nn.Conv2d(19,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,8,3,padding=1),nn.GroupNorm(2,8),nn.GELU())
        self.out=nn.Conv2d(8,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==68
        raw=self.raw(x[:,:1]);semantic=self.semantic(x[:,1:65]);correction=self.out(self.merge(torch.cat([raw,semantic,x[:,65:68]],dim=1)))
        prior=torch.logit(x[:,65:66].clamp(1e-5,1-1e-5))
        return prior+correction


def foreground_loss(logits,target):
    probability=logits.sigmoid()
    return F.binary_cross_entropy_with_logits(logits,target)+1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
