"""Two-parameter monotone foreground-logit adjustment; no threshold search."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import torch


def objective(parameters, logits, targets, weights):
    scale, bias = parameters
    scores = scale * logits + bias
    losses = np.logaddexp(0., scores) - targets * scores
    error = expit(scores) - targets
    return float(np.sum(weights * losses)), np.array([np.sum(weights * error * logits), np.sum(weights * error)])


def fit(probabilities, targets, groups):
    assert probabilities.shape == targets.shape and probabilities.ndim == 3
    probabilities = np.clip(probabilities.astype(np.float64), 1e-6, 1-1e-6)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    targets = targets.astype(np.float64)
    groups = np.asarray(groups); unique = np.unique(groups)
    weights = np.empty(probabilities.shape, np.float64)
    for group in unique:
        selected = groups == group
        weights[selected] = 1. / (len(unique) * selected.sum() * probabilities.shape[1] * probabilities.shape[2])
    result = minimize(objective, np.array([1., 0.]), args=(logits, targets, weights), jac=True,
                      method='L-BFGS-B', bounds=[(1e-6, None), (None, None)],
                      options=dict(maxiter=100, gtol=1e-8, ftol=1e-12))
    assert result.success and np.isfinite(result.x).all() and result.x[0] > 0
    initial, _ = objective(np.array([1., 0.]), logits, targets, weights)
    final, gradient = objective(result.x, logits, targets, weights)
    assert final <= initial + 1e-12
    scale, bias = map(float, result.x)
    return dict(scale=scale, bias=bias, equivalent_foreground_cutoff=float(expit(-bias/scale)),
                initial_nll=initial, fitted_nll=final, gradient=list(map(float, gradient)),
                iterations=int(result.nit), optimizer_message=str(result.message), success=bool(result.success),
                group_counts={str(g):int((groups==g).sum()) for g in unique})


class CalibratedForeground(torch.nn.Module):
    """Apply after D4 probability averaging, preserving the auxiliary channel."""
    def __init__(self, model, scale, bias):
        super().__init__(); self.model=model; self.scale=float(scale); self.bias=float(bias)
        assert self.scale > 0 and np.isfinite([self.scale, self.bias]).all()

    def forward(self, x):
        logits=self.model(x)
        return torch.cat([self.scale*logits[:,:1]+self.bias,logits[:,1:]],1)
