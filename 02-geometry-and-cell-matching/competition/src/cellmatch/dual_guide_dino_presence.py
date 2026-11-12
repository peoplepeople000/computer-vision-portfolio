"""Public CPDINO normalized-query patch tokens, not whole-native-image tokens."""
import torch
from torch.nn import functional as F


def encode_input(x):
    # Same global-normalized gray as595; no guide or GT injected into encoder.
    return F.interpolate(x[:, :1].float(), size=(128, 128), mode='bilinear', align_corners=False)


def pool_weights(x):
    native, public = x[:, 1:2].float(), x[:, 2:3].float()
    context = 1 - torch.maximum(native, public)
    planes = torch.cat([native, public, context], 1)
    weights = F.interpolate(planes, size=(16, 16), mode='area').flatten(2)
    assert torch.all(weights.sum(-1) > 0)
    return weights / weights.sum(-1, keepdim=True)


def pool_tokens(tokens, x):
    assert tokens.shape == (len(x), 256, 1024)
    weights = pool_weights(x).to(tokens.device)
    return torch.bmm(weights, tokens).flatten(1)
