"""Small task-specific partial point matcher inspired by SuperGlue (CVPR2020).

Locally authored prototype: dual input encoders, PyTorch attention, pointwise
LayerNorm and balanced log-domain transport. No official model/code/weights are
embedded. See https://arxiv.org/abs/1911.11763 for the research principle.
"""
import math

import torch
from torch import nn


def partial_log_assignment(scores, unmatched_score, iterations=50):
    """Unit real-node masses with multi-capacity unmatched row and column.

The extra source has mass equal to target count; the extra target has mass
equal to source count. Thus many unmatched real nodes can share a dustbin.
Returned probabilities are rescaled so each real row/column targets mass1.
"""
    n, m = scores.shape
    assert n > 0 and m > 0
    top = torch.cat((scores, unmatched_score.expand(n, 1)), dim=1)
    logits = torch.cat((top, unmatched_score.expand(1, m + 1)), dim=0)
    source_mass = scores.new_ones(n + 1)
    target_mass = scores.new_ones(m + 1)
    source_mass[-1] = m
    target_mass[-1] = n
    total = n + m
    log_source = (source_mass / total).log()
    log_target = (target_mass / total).log()
    source_dual, target_dual = torch.zeros_like(log_source), torch.zeros_like(log_target)
    for _ in range(iterations):
        source_dual = log_source - torch.logsumexp(logits + target_dual[None], dim=1)
        target_dual = log_target - torch.logsumexp(logits + source_dual[:, None], dim=0)
    return logits + source_dual[:, None] + target_dual[None] + math.log(total)


class ContextUpdate(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.update = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, query, context):
        message, _ = self.attention(self.norm(query)[None], self.norm(context)[None],
                                    self.norm(context)[None], need_weights=False)
        return query + self.update(torch.cat((query, message[0]), dim=-1))


class GraphPointMatcher(nn.Module):
    def __init__(self, width=64, rounds=2, iterations=50):
        super().__init__()
        self.width, self.iterations = width, iterations
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(32, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width)) for _ in range(2)])
        self.self_updates = nn.ModuleList([ContextUpdate(width) for _ in range(rounds)])
        self.cross_updates = nn.ModuleList([ContextUpdate(width) for _ in range(rounds)])
        self.projection = nn.Linear(width, width)
        self.unmatched_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, iv, ex):
        a, b = self.encoders[0](iv), self.encoders[1](ex)
        for self_layer, cross_layer in zip(self.self_updates, self.cross_updates):
            a, b = self_layer(a, a), self_layer(b, b)
            a, b = cross_layer(a, b), cross_layer(b, a)
        a, b = self.projection(a), self.projection(b)
        logits = a @ b.T / math.sqrt(self.width)
        return partial_log_assignment(logits, self.unmatched_score, self.iterations)


def assignment_loss(log_probabilities, edges):
    n, m = log_probabilities.shape[0] - 1, log_probabilities.shape[1] - 1
    assert len(edges) > 0
    positive = -log_probabilities[edges[:, 0], edges[:, 1]].mean()
    source_free = torch.ones(n, dtype=torch.bool, device=log_probabilities.device)
    target_free = torch.ones(m, dtype=torch.bool, device=log_probabilities.device)
    source_free[edges[:, 0]] = False
    target_free[edges[:, 1]] = False
    negatives = torch.cat((log_probabilities[:-1, -1][source_free],
                           log_probabilities[-1, :-1][target_free]))
    assert len(negatives) > 0
    unmatched = -negatives.mean()
    return (positive + unmatched) / 2, positive, unmatched


def decode_assignment(log_probabilities, threshold=.2):
    """Mutual full-row/full-column winners, including unmatched alternatives."""
    probability = log_probabilities.exp()
    n, m = probability.shape[0] - 1, probability.shape[1] - 1
    target = probability[:-1].argmax(dim=1)
    source = probability[:, :-1].argmax(dim=0)
    result = []
    for i, j in enumerate(target.tolist()):
        if j < m and int(source[j]) == i and float(probability[i, j]) > threshold:
            result.append((i, j))
    return result
