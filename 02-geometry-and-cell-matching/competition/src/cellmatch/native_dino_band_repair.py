"""Conditional contour residual with a frozen spatial CPDINO token input."""
import torch
from torch import nn
from torch.nn import functional as F
from .dual_guide_object_repair import DualGuideObjectRepair


def token_grid(tokens):
    assert tokens.ndim == 3 and tokens.shape[1:] == (256, 1024)
    return tokens.reshape(len(tokens),16,16,1024).permute(0,3,1,2).contiguous()


def normalized_grid(grid):
    assert grid.shape[1:] == (1024,16,16)
    return F.layer_norm(grid.permute(0,2,3,1),(1024,),eps=1e-5).permute(0,3,1,2)


class DinoGridBandRepair(DualGuideObjectRepair):
    def __init__(self):
        super().__init__()
        # Keep inherited initialization and the subsequent Torch RNG unchanged.
        state = torch.get_rng_state()
        self.spatial = nn.Conv2d(1024,64,1,bias=False)
        nn.init.zeros_(self.spatial.weight)
        torch.set_rng_state(state)

    def forward(self,x,grid):
        assert grid.shape == (len(x),1024,16,16) and x.shape[1:] == (3,64,64)
        gray=x[:,:1]
        mean=gray.mean((2,3),keepdim=True)
        sd=(gray-mean).square().mean((2,3),keepdim=True).sqrt().clamp_min(.02)
        gray=((gray-mean)/sd).clamp(-3,3)/3
        a=self.enc1(torch.cat([gray,x[:,1:]],1))
        b=self.enc2(F.max_pool2d(a,2))
        c=self.enc3(F.max_pool2d(b,2)) + self.spatial(normalized_grid(grid))
        pooled=torch.cat([c.mean((2,3)),c.amax((2,3))],1)
        presence=self.presence(pooled)[:,0]
        d=self.dec2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        contour=(2*x[:,1:2]-1)*2.1972245773362196+self.contour_delta(e)
        return contour,presence
