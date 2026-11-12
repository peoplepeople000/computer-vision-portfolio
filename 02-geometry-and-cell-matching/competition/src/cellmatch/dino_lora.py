"""Small additive adapters preserving the original DINO linear forward path."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=8.):
        super().__init__()
        if not isinstance(base,nn.Linear):raise TypeError('Expected a linear projection')
        self.base=base.requires_grad_(False);self.in_features=base.in_features;self.out_features=base.out_features
        self.scale=float(alpha)/rank
        self.a=nn.Parameter(base.weight.new_empty((rank,self.in_features)))
        self.b=nn.Parameter(base.weight.new_zeros((self.out_features,rank)))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.a),self.b)*self.scale


def install(net,rank=8,alpha=8.):
    net.requires_grad_(False);names=[]
    for i,block in enumerate(net.encoder.blocks):
        for name in ['qkv','proj']:
            base=getattr(block.attn,name)
            setattr(block.attn,name,LoRALinear(base,rank,alpha));names.append(f'encoder.blocks.{i}.attn.{name}')
    net.out.requires_grad_(True)
    return names


def adapter_state(net):
    return {name:p.detach().cpu().clone() for name,p in net.named_parameters() if p.requires_grad}


def load_adapter(net,state):
    params={name:p for name,p in net.named_parameters() if p.requires_grad}
    if set(params)!=set(state):raise ValueError('Adapter parameter names differ')
    with torch.no_grad():
        for name,p in params.items():
            if p.shape!=state[name].shape:raise ValueError('Adapter shape differs')
            p.copy_(state[name])
