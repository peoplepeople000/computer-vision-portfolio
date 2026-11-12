"""Requested prior-conditioned cyto3 initialization, pending CPU preflight."""
import torch
from torch import nn
from torch.nn import functional as F
from cellpose.resnet_torch import CPnet

class ConditionalCytoRefiner(nn.Module):
 def __init__(self,weights):
  super().__init__();self.net=CPnet([2,32,64,128,256],3,3,mkldnn=False)
  self.net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
  first=self.net.downsample.down[0];self.zeroed=[]
  for name,layer in first.named_modules():
   if isinstance(layer,nn.Conv2d) and layer.in_channels==2:
    with torch.no_grad():layer.weight[:,1].zero_()
    self.zeroed.append(name)
  assert len(self.zeroed)==2
  for parameter in self.parameters():parameter.requires_grad_(True)
 def enlarged(self,x):
  raw=F.interpolate(x[:,:1],size=(96,96),mode='bilinear',align_corners=False)
  prior=F.interpolate(x[:,2:3],size=(96,96),mode='nearest')
  return torch.cat([raw,prior],1)
 def forward(self,x):return self.net(self.enlarged(x))[0]
 def native_fg(self,x):return F.avg_pool2d(self.forward(x)[:,2:3],3,3)
