"""Exact four-phase native bilinear sampling for a frozen 1x1 readout."""
import torch


def phases(features):
    assert features.ndim==4 and features.shape[0]==1
    assert features.shape[-2]%4==features.shape[-1]%4==0
    return torch.cat([features[...,y::4,x::4] for y,x in ((1,1),(1,2),(2,1),(2,2))],dim=0)


def native_readout(head,features):
    assert features.shape[0]==4 and head.kernel_size==(1,1)
    return head(features).sigmoid().mean(dim=0,keepdim=True)
