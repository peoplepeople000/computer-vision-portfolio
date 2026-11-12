"""Training-derived per-instance spatial normalization with native-grid output."""
import torch
import torch.nn.functional as F
from .learning import SmallUNet

def sample_border(x,grid):
    # Clamping to boundary pixel centers exactly implements border padding,
    # including on MPS versions that reject padding_mode='border'.
    h,w=x.shape[-2:]
    limit=grid.new_tensor([1-1/w,1-1/h])
    grid=torch.maximum(torch.minimum(grid,limit),-limit)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)

class CanonicalNativeRefiner(torch.nn.Module):
    def __init__(self, area):
        super().__init__()
        self.net=SmallUNet(16)
        self.register_buffer('canonical_area',torch.tensor(float(area)))

    def grids(self,x):
        n,_,h,w=x.shape
        prior=x[:,2:3].clamp(0,1)
        area=prior.sum((1,2,3)).clamp_min(1)
        scale=(self.canonical_area/area).sqrt().clamp(.5,2.)
        xx=(torch.arange(w,device=x.device,dtype=x.dtype)+.5)*2/w-1
        yy=(torch.arange(h,device=x.device,dtype=x.dtype)+.5)*2/h-1
        cx=(prior*xx[None,None,None,:]).sum((1,2,3))/area
        cy=(prior*yy[None,None,:,None]).sum((1,2,3))/area
        theta=x.new_zeros(n,2,3);theta[:,0,0]=1/scale;theta[:,1,1]=1/scale
        theta[:,0,2]=cx;theta[:,1,2]=cy
        inverse=x.new_zeros(n,2,3);inverse[:,0,0]=scale;inverse[:,1,1]=scale
        inverse[:,0,2]=-scale*cx;inverse[:,1,2]=-scale*cy
        return F.affine_grid(theta,x.shape,align_corners=False),F.affine_grid(inverse,(n,2,h,w),align_corners=False)

    def forward(self,x):
        grid,inverse=self.grids(x)
        canonical=sample_border(x,grid)
        logits=self.net(canonical)
        restored=sample_border(logits,inverse)
        valid=(inverse.abs()<=1).all(-1)[:,None]
        fallback=torch.cat([(x[:,2:3]-.5)*16,torch.full_like(x[:,2:3],-8)],dim=1)
        return torch.where(valid,restored,fallback)
