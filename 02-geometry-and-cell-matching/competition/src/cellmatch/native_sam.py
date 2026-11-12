"""Detached first-order SAM weight perturbation for the native contour task.

Independent implementation of Foret et al. SAM's normalized-gradient step.
Decoupled AdamW decay remains outside the task loss and inner perturbation.
"""
import torch


@torch.no_grad()
def perturb(parameters,rho=.05):
    parameters=list(parameters)
    active=[p for p in parameters if p.grad is not None]
    assert active
    norm=torch.stack([p.grad.detach().norm(2) for p in active]).norm(2)
    assert torch.isfinite(norm)
    originals=[p.detach().clone() for p in active]
    scale=rho/norm.clamp_min(1e-12)
    for p in active:p.add_(p.grad.detach()*scale)
    actual=torch.stack([(p-v).norm(2) for p,v in zip(active,originals)]).norm(2)
    return active,originals,dict(gradient_norm=float(norm.cpu()),actual_perturbation_norm=float(actual.cpu()))


@torch.no_grad()
def restore(parameters,originals):
    assert len(parameters)==len(originals)
    for p,v in zip(parameters,originals):p.copy_(v)
