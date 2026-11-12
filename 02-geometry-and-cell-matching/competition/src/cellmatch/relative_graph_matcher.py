"""Locally authored relative-distance point-bank attention prototype.

The geometric-conditioning principle is related to GeoTransformer CVPR2022
(https://arxiv.org/abs/2202.06688). This is not its implementation, model, or
distance-and-angle encoding. Native-pixel radial biases are rigid-frame
invariant, but not generally affine invariant.
"""
import math
import torch
from torch import nn
from .graph_point_matcher import ContextUpdate, partial_log_assignment


def distance_basis(points):
    distances = torch.cdist(points, points, compute_mode='donot_use_mm_for_euclid_dist')
    centers = torch.logspace(math.log10(8), math.log10(256), 16,
                             device=points.device, dtype=points.dtype)
    return torch.exp(-.5*((torch.log1p(distances)[...,None]-torch.log1p(centers))/.5)**2)


class RelativeSelfUpdate(ContextUpdate):
    def __init__(self, width=64):
        super().__init__(width)
        self.radial_bias = nn.Linear(16, 4, bias=False)

    def forward(self, query, basis):
        values = self.norm(query)[None]
        bias = self.radial_bias(basis).permute(2,0,1).contiguous()
        message, _ = self.attention(values, values, values, attn_mask=bias, need_weights=False)
        return query + self.update(torch.cat((query,message[0]),dim=-1))


class RelativeGraphMatcher(nn.Module):
    def __init__(self, width=64, rounds=2, iterations=50):
        super().__init__()
        self.width, self.iterations = width, iterations
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(30,width),nn.SiLU(),
            nn.Linear(width,width),nn.LayerNorm(width)) for _ in range(2)])
        self.self_updates = nn.ModuleList([RelativeSelfUpdate(width) for _ in range(rounds)])
        self.cross_updates = nn.ModuleList([ContextUpdate(width) for _ in range(rounds)])
        self.projection = nn.Linear(width,width)
        self.unmatched_score = nn.Parameter(torch.tensor(1.))

    def forward(self, iv, ex, iv_points, ex_points):
        a, b = self.encoders[0](iv), self.encoders[1](ex)
        a_basis, b_basis = distance_basis(iv_points), distance_basis(ex_points)
        for self_layer, cross_layer in zip(self.self_updates,self.cross_updates):
            a, b = self_layer(a,a_basis), self_layer(b,b_basis)
            a, b = cross_layer(a,b), cross_layer(b,a)
        a, b = self.projection(a), self.projection(b)
        return partial_log_assignment(a@b.T/math.sqrt(self.width), self.unmatched_score,self.iterations)
