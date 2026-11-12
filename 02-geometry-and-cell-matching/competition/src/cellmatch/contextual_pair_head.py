"""Compact residual attention over alternative paired contours.

This is a custom candidate scorer, not SuperGlue. Existing endpoint gates and
optional unmatched selection stay external. Every IV family has unit attention
mass before content/spatial weighting; duplicate edges cannot multiply support.
"""
import copy
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn
from .rle import Mask


def attention_inputs(record,matrix,shape):
    candidates=record['candidates'];unique=defaultdict(set);counts=Counter()
    for c in candidates:
        key=(c['family'],c['iv_rle'],c['ex_rle']);counts[key]+=1;unique[c['family']].add(key)
    positions=[];weights=[];centers={};matrix=np.asarray(matrix)
    for c in candidates:
        rr=c['iv_rle']
        if rr not in centers:centers[rr]=Mask.parse(rr,shape).coordinates().mean(0)
        positions.append(matrix[:,:2]@centers[rr]+matrix[:,2]);key=(c['family'],rr,c['ex_rle']);weights.append(1./(len(unique[c['family']])*counts[key]))
    return np.asarray(positions,np.float32).reshape(-1,2),np.asarray(weights,np.float32)


class ContextualPairHead(nn.Module):
    def __init__(self,linear,width=32):
        super().__init__();self.baseline=copy.deepcopy(linear)
        for p in self.baseline.parameters():p.requires_grad_(False)
        self.embed=nn.Sequential(nn.Linear(linear.in_features,width),nn.SiLU())
        self.query=nn.Linear(width,width,bias=False);self.key=nn.Linear(width,width,bias=False);self.value=nn.Linear(width,width,bias=False)
        self.residual=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.width=width

    def forward(self,x,positions,weights):
        original=self.baseline(x).ravel()
        if not len(x):return original
        h=self.embed(x);q=self.query(h);k=self.key(h);v=self.value(h)
        delta=(positions[:,None,:]-positions[None,:,:])/128.
        spatial=-.5*delta.square().sum(-1)
        logits=(q@k.T)/(self.width**.5)+spatial+weights.log()[None,:]
        context=logits.softmax(-1)@v
        return original+self.residual(torch.cat([h,context],-1)).ravel()
