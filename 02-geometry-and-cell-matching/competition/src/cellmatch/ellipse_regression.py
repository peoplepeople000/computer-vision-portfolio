"""Bounded image-conditioned center/area correction with fixed moment aspect."""
import numpy as np
import torch
from torch import nn
from .guided_ex_data import cut

LOG_AREA_BOUND = float(np.log(1.5))


def parameters(xy):
    xy=np.asarray(xy,np.float64);center=xy.mean(0);z=xy-center
    cov=z.T@z/len(xy)+np.eye(2)/12
    return center,float(len(xy)),cov/np.sqrt(np.linalg.det(cov))


def raster(center,area,aspect,shape):
    matrix=aspect*area/np.pi;radius=np.sqrt(np.diag(matrix))
    lo=np.maximum(np.floor(center-radius).astype(int),0)
    hi=np.minimum(np.ceil(center+radius).astype(int),np.array(shape[::-1])-1)
    xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
    xy=np.c_[xx.ravel(),yy.ravel()];z=xy-center
    return xy[np.einsum('ni,ij,nj->n',z,np.linalg.inv(matrix),z)<=1]


def corrected(xy,normalized,shape):
    center,area,aspect=parameters(xy);value=np.asarray(normalized,np.float64)
    assert np.linalg.norm(value[:2])<=1.00001 and abs(value[2])<=1.00001
    return raster(center+2*value[:2],area*np.exp(LOG_AREA_BOUND*value[2]),aspect,shape)


def crop(features,xy,size=48):
    xy=np.asarray(xy);origin=np.rint(xy.mean(0)).astype(int)-size//2
    local=xy-origin;valid=((local>=0)&(local<size)).all(1)
    mask=np.zeros((1,size,size),np.float32);mask[0,local[valid,1],local[valid,0]]=1
    return np.concatenate([cut(features,*map(int,origin),size),mask]),int((~valid).sum())


def target(xy,gt):
    center,area,_=parameters(xy);gc,ga,_=parameters(gt)
    delta=(gc-center)/2;delta/=max(1.,float(np.linalg.norm(delta)))
    return np.r_[delta,np.clip(np.log(ga/area)/LOG_AREA_BOUND,-1,1)].astype(np.float32)


def augment(x,y,rotation,reflected):
    y=y.clone()
    if reflected:
        x=x.flip(-1);y[:,0]=-y[:,0]
    x=torch.rot90(x,rotation,(-2,-1))
    for _ in range(rotation):y=torch.stack([y[:,1],-y[:,0],y[:,2]],1)
    return x.contiguous(),y


class EllipseRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.AvgPool2d(2),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten())
        self.head=nn.Linear(512,3);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        raw=self.head(self.tower(x)).tanh()
        center=raw[:,:2]/raw[:,:2].norm(dim=1,keepdim=True).clamp_min(1.)
        return torch.cat([center,raw[:,2:]],1)
