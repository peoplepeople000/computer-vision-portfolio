"""Joint pair-credit classifier with an additional within-image ranking loss."""
import numpy as np
import torch
from torch import nn


def ranking_loss(logits, groups):
    terms = []
    for positive, negative in groups:
        terms.append(nn.functional.softplus(logits[negative][None, :] - logits[positive][:, None]).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def fit(records, ranking_weight=1.):
    x = np.concatenate([r['x'] for r in records])
    y = np.concatenate([r['y'] for r in records]).astype(np.float32)
    mean = x.mean(0); sd = np.maximum(x.std(0), .03)
    xt = torch.from_numpy(((x-mean)/sd).clip(-8, 8)); yt = torch.from_numpy(y)
    groups = []; start = 0
    for r in records:
        labels = np.asarray(r['y'], bool)
        positive = np.flatnonzero(labels)+start; negative = np.flatnonzero(~labels)+start
        if len(positive) and len(negative):
            groups.append((torch.tensor(positive), torch.tensor(negative)))
        start += len(labels)
    torch.manual_seed(2033); model = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=150, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(); logits = model(xt).ravel()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)+.02*model.weight.square().mean()
        if ranking_weight:
            loss = loss+ranking_weight*ranking_loss(logits, groups)
        loss.backward(); return loss
    optimizer.step(closure)
    return model.eval(), mean, sd
