"""Frozen native spatial features with a zero-initialized foreground correction."""
import torch
from torch import nn
from torch.nn import functional as F


class ForegroundAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Conv2d(16,16,3,padding=1)
        self.output = nn.Conv2d(16,1,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(F.silu(self.hidden(features)))


def features_and_logits(model, x):
    a=model.a(x);b=model.b(F.avg_pool2d(a,2));c=model.c(F.avg_pool2d(b,2))
    d=model.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
    features=model.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
    return features, model.out(features)


def d4_cached_probability(adapter, features, logits):
    """Inputs already inverted into native coordinates, including feature channels."""
    shape=features.shape
    # Features were spatially inverted before caching. The learned 3x3 kernel
    # must operate in each model-view coordinate system, then invert its output.
    probabilities=[]
    for view in range(8):
        reflected=view//4; turns=view%4
        native=features[:,view]
        transformed=torch.rot90(native.flip(-1) if reflected else native,turns,(-2,-1))
        delta=adapter(transformed)
        delta=torch.rot90(delta,-turns,(-2,-1))
        if reflected:delta=delta.flip(-1)
        probabilities.append((logits[:,view,:1]+delta).sigmoid())
    return torch.stack(probabilities).mean(0)


def adaptation_loss(probability, target, teacher, preserve):
    """Repair GT BCE+Dice, preserve teacher Bernoulli KL; fixed cohort labels."""
    q=probability.clamp(1e-6,1-1e-6);p=teacher.clamp(1e-6,1-1e-6)
    fg=target
    bce=-(2*fg*q.log()+(1-fg)*torch.log1p(-q)).mean((1,2,3))
    dice=1-(2*(q*fg).sum((1,2,3))+1)/(q.sum((1,2,3))+fg.sum((1,2,3))+1)
    kl=p*(p.log()-q.log())+(1-p)*(torch.log1p(-p)-torch.log1p(-q))
    weight=1+4*p
    preservation=(kl*weight).sum((1,2,3))/weight.sum((1,2,3))
    return torch.where(preserve,preservation,bce+dice).mean()
