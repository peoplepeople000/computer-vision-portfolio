"""Equal training loss mass per IV family, retaining original normalization."""
from collections import Counter
import numpy as np
import torch
from torch import nn


def weights(record):
    counts = Counter(c['family'] for c in record['candidates'])
    return np.asarray([1. / counts[c['family']] for c in record['candidates']], np.float32)


def fit(records):
    x = np.concatenate([r['x'] for r in records])
    y = np.concatenate([r['y'] for r in records]).astype(np.float32)
    w = np.concatenate([weights(r) for r in records])
    w /= w.mean()
    mean = x.mean(0)
    sd = np.maximum(x.std(0), .03)
    xt = torch.from_numpy(((x - mean) / sd).clip(-8, 8))
    yt, wt = torch.from_numpy(y), torch.from_numpy(w)
    torch.manual_seed(2033)
    model = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=150, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        bce = nn.functional.binary_cross_entropy_with_logits(model(xt).ravel(), yt, reduction='none')
        loss = (bce * wt).mean() + .02 * model.weight.square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return model.eval(), mean, sd
