"""Frozen public Cellpose3 encoder moments for native EX annotation presence."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cellpose.resnet_torch import CPnet
from .native_presence_embedding import crops


class CytoPresenceEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.encoder = CPnet([2, 32, 64, 128, 256], 3, 3, mkldnn=False)
        self.encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        # Official grayscale convention: segmenting channel plus zero nuclear channel.
        grayscale = torch.cat([x[:, :1], torch.zeros_like(x[:, :1])], 1)
        grayscale = F.interpolate(grayscale, scale_factor=2., mode='bilinear', align_corners=False)
        parts = []
        for z in self.encoder.downsample(grayscale):
            prior = F.interpolate(x[:, 2:3], size=z.shape[-2:], mode='area')
            for weights in [prior, 1-prior]:
                parts.append((z*weights).sum((2, 3))/weights.sum((2, 3)).clamp_min(1e-6))
        return torch.cat(parts, 1)


def embed_d4(model, chips, device, batch=64):
    output = []
    with torch.inference_mode():
        for start in range(0, len(chips), batch):
            x = torch.from_numpy(chips[start:start+batch].astype(np.float32)).to(device)
            values = []
            for flip in [False, True]:
                for turns in range(4):
                    view = torch.rot90(x, turns, (2, 3))
                    if flip: view = view.flip(3)
                    values.append(model(view).cpu().numpy())
            output.append(np.mean(values, axis=0))
    result = np.concatenate(output)
    assert result.shape == (len(chips), 960) and np.isfinite(result).all()
    return result.astype(np.float32)
