"""Binary Lovasz hinge foreground plus unchanged official Cellpose flow loss.
Lovasz formula: Berman et al., CVPR2018, https://arxiv.org/abs/1705.08790.
Adapted from Maxim Berman's MIT-licensed reference; license retained at
vendor/lovasz-reference/LICENSE. Copyright (c) 2018 Maxim Berman.
"""
import torch
from torch.nn import functional as F


def lovasz_hinge(logits, labels):
    if logits.shape != labels.shape or logits.ndim != 3:
        raise ValueError('Expected equal BHW logits and binary targets')
    if not logits.numel():
        return logits.sum()*0
    values=[]
    for z,y in zip(logits,labels):
        z=z.flatten();y=y.flatten().to(z.dtype)
        errors=1-z*(2*y-1)
        errors,order=errors.sort(descending=True)
        ranked=y[order];positives=ranked.sum()
        jaccard=1-(positives-ranked.cumsum(0))/(positives+(1-ranked).cumsum(0))
        weights=torch.cat((jaccard[:1],jaccard[1:]-jaccard[:-1]))
        values.append(torch.dot(errors.relu(),weights))
    return torch.stack(values).mean()


def loss_fn_seg(labels,prediction,device=None):
    flow=F.mse_loss(prediction[:,-3:-1],5*labels[:,-2:])/2
    foreground=lovasz_hinge(prediction[:,-1],labels[:,-3]>.5)
    return flow+foreground
