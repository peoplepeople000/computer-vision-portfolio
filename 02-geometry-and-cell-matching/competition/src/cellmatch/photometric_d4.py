"""Fixed photometric probability ensemble; query prior stays unchanged."""
import torch
from .ex_local_boundary import D4Average


class PhotometricD4(torch.nn.Module):
    def __init__(self, model, gammas=(0.8, 1.0, 1.25)):
        super().__init__()
        self.spatial = D4Average(model)
        self.gammas = tuple(gammas)

    def forward(self, x):
        if self.gammas == (1.0,):
            return self.spatial(x)
        probabilities = []
        for gamma in self.gammas:
            view = x.clone()
            view[:, :2] = view[:, :2].clamp(0, 1).pow(gamma)
            probabilities.append(self.spatial(view).sigmoid())
        return torch.logit(torch.stack(probabilities).mean(0).clamp(1e-6, 1-1e-6))
