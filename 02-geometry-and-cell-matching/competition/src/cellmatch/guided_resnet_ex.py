"""Prompt-conditioned ResNet EX with exact zero-adapter initialization parity."""
import torch
from torch import nn
from torch.nn import functional as F
from .resnet_ex import ResNetEX

class GuidedResNetEX(ResNetEX):
 def __init__(self,initial_state=None):
  super().__init__()
  if initial_state is not None:self.load_state_dict(initial_state)
  self.prompt=nn.Conv2d(4,64,3,padding=1,bias=False)
  nn.init.zeros_(self.prompt.weight)
  self.presence=nn.Linear(512,1);nn.init.zeros_(self.presence.weight);nn.init.zeros_(self.presence.bias)

 def forward(self,x):
  if x.ndim!=4 or x.shape[1]!=9:raise ValueError('Expected Bx9xHxW guided inputs')
  rgb=torch.cat([x[:,:1],x[:,:1],x[:,1:2]],1)
  rgb=F.interpolate(rgb,scale_factor=2.,mode='bilinear',align_corners=False);rgb=(rgb-self.mean)/self.std;e=self.encoder
  a=e.relu(e.bn1(e.conv1(rgb)));prompt=torch.cat([x[:,3:5],x[:,7:9]],1);a=a+self.prompt(F.interpolate(prompt,size=a.shape[-2:],mode='bilinear',align_corners=False))
  b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);f=e.layer4(d)
  def up(z,skip,block):return block(torch.cat([F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
  z=up(f,d,self.d3);z=up(z,c,self.d2);z=up(z,b,self.d1);z=up(z,a,self.d0)
  return F.interpolate(self.out(z),size=x.shape[-2:],mode='bilinear',align_corners=False),self.presence(f.mean((2,3))).ravel()
