"""Fixed prior-centroid conditional polar boundary; Stage397 targets unchanged."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .conditional_polar import prior_center, radial_target, ANGLES, STEP, MAX_RADIUS


def render(radii, center, shape=(32,32)):
    y,x=np.indices(shape);x=x-center[0];y=y-center[1]
    angle=np.mod(np.arctan2(y,x),2*np.pi)*ANGLES/(2*np.pi)
    lower=np.floor(angle).astype(int)%ANGLES;fraction=angle-np.floor(angle)
    boundary=radii[lower]*(1-fraction)+radii[(lower+1)%ANGLES]*fraction
    return (boundary>0)&(np.hypot(x,y)<=boundary)


def labels_from_radii(radii):
    # np.rint implements the specified round, including ties-to-even at cap24.
    labels=np.where(radii==0,0,np.rint(radii/STEP+.5)).astype(np.int64)
    assert labels.min()>=0 and labels.max()<=96
    return labels


def d4_radii(radii,k,flip):
    """Cartesian np.rot90(k), followed by horizontal flip; y increases down."""
    old=np.arange(ANGLES);new=(old-16*k)%ANGLES
    if flip:new=(32-new)%ANGLES
    result=np.empty_like(radii);result[...,new]=radii[...,old]
    return result


class CircularConv(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.conv=nn.Conv2d(cin,cout,3)
    def forward(self,x):
        x=F.pad(x,(0,0,1,1),mode='circular')
        return F.silu(self.conv(F.pad(x,(1,1,0,0),mode='constant',value=0)))


class PolarBoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(CircularConv(4,16),CircularConv(16,16),CircularConv(16,16),nn.Conv2d(16,1,1))
        theta=torch.arange(64,dtype=torch.float32)*2*torch.pi/64
        radius=torch.arange(97,dtype=torch.float32)*.25
        self.register_buffer('offsets',torch.stack([theta.cos()[:,None]*radius,theta.sin()[:,None]*radius],-1))
        self.register_buffer('radius_coordinate',(radius/24)[None,None,None,:])
        self.register_buffer('class_radius',torch.cat([torch.zeros(1),torch.arange(96,dtype=torch.float32)*.25+.125]))
        yy,xx=torch.meshgrid(torch.arange(32,dtype=torch.float32),torch.arange(32,dtype=torch.float32),indexing='ij')
        self.register_buffer('pixels',torch.stack([xx,yy],-1))
    def centers(self,x):
        mask=(x[:,2]>.5).to(x.dtype);area=mask.sum((1,2));
        # Inputs are original nonempty native priors; never substitute a target center.
        return (mask[...,None]*self.pixels).sum((1,2))/area[:,None]
    def polar_inputs(self,x):
        center=self.centers(x)
        grid=(center[:,None,None,:]+self.offsets[None])/31*2-1
        sampled=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        coordinate=self.radius_coordinate.expand(len(x),1,64,97)
        return torch.cat([sampled,coordinate],1)
    def boundary_logits(self,x):return self.layers(self.polar_inputs(x))[:,0]
    def expected_radii(self,logits):return (logits.softmax(-1)*self.class_radius).sum(-1)
    def foreground_logits(self,radii,center):
        delta=self.pixels[None]-center[:,None,None,:]
        angular=torch.remainder(torch.atan2(delta[...,1],delta[...,0]),2*torch.pi)*64/(2*torch.pi)
        lower=angular.floor().long()%64;fraction=angular-angular.floor()
        lo=radii.gather(1,lower.flatten(1)).reshape_as(angular)
        hi=radii.gather(1,((lower+1)%64).flatten(1)).reshape_as(angular)
        boundary=lo*(1-fraction)+hi*fraction
        distance=delta.square().sum(-1).sqrt()
        logits=2*(boundary-distance)
        # A radius0 boundary is empty, including distance0: finite negative logit.
        logits=torch.where(boundary>0,logits,torch.full_like(logits,-1.))
        return logits[:,None]
    def forward(self,x):
        return self.foreground_logits(self.expected_radii(self.boundary_logits(x)),self.centers(x))
