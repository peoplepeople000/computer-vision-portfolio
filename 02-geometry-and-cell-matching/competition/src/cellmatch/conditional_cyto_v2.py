"""Public cyto3 plus zero-initialized trainable prior-to-input projection."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cellpose.resnet_torch import CPnet

class ConditionalCytoV2(nn.Module):
 def __init__(self,weights):
  super().__init__();self.net=CPnet([2,32,64,128,256],3,3,mkldnn=False)
  self.net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
  self.prior_projection=nn.Conv2d(1,2,1,bias=False)
  nn.init.zeros_(self.prior_projection.weight)
 def forward(self,x):
  raw=F.interpolate(x[:,:1],size=(96,96),mode='bilinear',align_corners=False)
  prior=F.interpolate(x[:,2:3],size=(96,96),mode='nearest')
  return self.net(torch.cat([raw,torch.zeros_like(raw)],1)+self.prior_projection(prior))[0]

class NativeFG(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):return F.avg_pool2d(self.model(x)[:,2:3],3,3)

def d4_labels(labels,turns,flip):
 """FG,flowY,flowX; reflect x first, then numpy CCW spatial rotation."""
 a=labels.copy()
 if flip:a=a[:,:,::-1].copy();a[2]*=-1
 a=np.rot90(a,turns,(1,2)).copy()
 for _ in range(turns):a[1],a[2]=-a[2].copy(),a[1].copy()
 return a

def augmented_batch(groups,subjects,rng,batch=8):
 xs=[];labels=[]
 for _ in range(batch):
  examples=groups[subjects[int(rng.integers(len(subjects)))]];xx,yy=examples[int(rng.integers(len(examples)))];k=int(rng.integers(4));flip=rng.random()<.5
  x=xx[:,:,::-1] if flip else xx;x=np.rot90(x,k,(1,2)).copy();x[0]=np.clip(x[0]**rng.uniform(.8,1.3)*rng.uniform(.85,1.15),0,1)
  xs.append(x);labels.append(d4_labels(yy,k,flip))
 return np.stack(xs).astype(np.float32),np.stack(labels).astype(np.float32)
