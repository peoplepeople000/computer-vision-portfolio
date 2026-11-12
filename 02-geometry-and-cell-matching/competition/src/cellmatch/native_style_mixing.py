"""Training-only cross-mouse feature-moment mixing for the native EX refiner.

Independent implementation of the MixStyle moment interpolation concept:
https://arxiv.org/abs/2104.02008
Unlike the author's batch-half swapping, donors are sampled from actual other
mouse identities in each batch. NumPy RNG isolates style draws from sampling.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .learning import SmallUNet


def mix_moments(x, donors, weights):
    mean = x.mean((2, 3), keepdim=True).detach()
    sd = (x.var((2, 3), keepdim=True, unbiased=True)+1e-6).sqrt().detach()
    donor = torch.as_tensor(donors, device=x.device, dtype=torch.long)
    weight = torch.as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
    mixed_mean = weight*mean+(1-weight)*mean[donor]
    mixed_sd = weight*sd+(1-weight)*sd[donor]
    return (x-mean)/sd*mixed_sd+mixed_mean


class StyleUNet(SmallUNet):
    def __init__(self, width=16, seed=3346):
        super().__init__(width)
        self.style_rng = np.random.default_rng(seed)
        self.domains = None
        self.style_counts = dict(a=0, b=0, donor_checks=0, single_domain_skips=0)

    def mix(self, x, layer):
        if not self.training:
            return x
        assert self.domains is not None and len(self.domains) == len(x)
        if self.style_rng.random() >= .5:
            return x
        domains = np.array(self.domains)
        if len(set(domains)) < 2:
            self.style_counts['single_domain_skips'] += 1
            return x
        donors = np.array([self.style_rng.choice(np.flatnonzero(domains != subject)) for subject in domains])
        assert np.all(domains[donors] != domains)
        weights = self.style_rng.beta(.1, .1, len(x)).astype(np.float32)
        self.style_counts[layer] += 1
        self.style_counts['donor_checks'] += len(x)
        return mix_moments(x, donors, weights)

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        a = self.mix(self.a(x), 'a')
        b = self.mix(self.b(F.avg_pool2d(a, 2)), 'b')
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        return self.out(self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1)))
