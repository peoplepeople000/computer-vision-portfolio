"""Task adaptation before pooling, with an exact frozen-prefix token cache."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


class QueryValueAdapter(nn.Module):
    """Keep the original (possibly masked-bias) QKV layer fully intact."""
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        assert base.out_features == 3*base.in_features
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha/rank
        device = base.weight.device
        for name in ['Aq', 'Av']:
            value = torch.empty(rank, self.in_features, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(value, a=5**.5)
            self.register_parameter(name, nn.Parameter(value))
        for name in ['Bq', 'Bv']:
            self.register_parameter(name, nn.Parameter(torch.zeros(self.in_features, rank, device=device, dtype=torch.float32)))

    def forward(self, x):
        original = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            q = F.linear(F.linear(value, self.Aq), self.Bq)*self.scale
            v = F.linear(F.linear(value, self.Av), self.Bv)*self.scale
            update = torch.cat([q, torch.zeros_like(q), v], dim=-1)
        return original+update.to(original.dtype)


def inject(model, count=8, rank=8, alpha=8):
    model.requires_grad_(False).eval()
    start = len(model.blocks)-count
    assert start >= 0
    for i in range(start, len(model.blocks)):
        base = model.blocks[i].attn.qkv
        assert not isinstance(base, QueryValueAdapter)
        model.blocks[i].attn.qkv = QueryValueAdapter(base, rank, alpha)
    return start


def adapter_parameters(model):
    return {n: p for n, p in model.named_parameters() if n.rsplit('.', 1)[-1] in ['Aq', 'Bq', 'Av', 'Bv']}


def base_hash(model):
    """Wrapper-independent, exact byte digest including pretrained buffers."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.rsplit('.', 1)[-1] in ['Aq', 'Bq', 'Av', 'Bv']:
            continue
        canonical = name.replace('qkv.base.', 'qkv.')
        h.update(canonical.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def prefix(model, images, start):
    """Only call in eval/no-grad mode; the cache is a neural input, not an image."""
    assert not model.training
    tokens, (height, width) = model.prepare_tokens_with_masks(images)
    rope = model.rope_embed(H=height, W=width) if model.rope_embed is not None else None
    for block in model.blocks[:start]:
        tokens = block(tokens, rope)
    return tokens, (height, width), rope


def suffix(model, tokens, hw, rope, start):
    for block in model.blocks[start:]:
        tokens = block(tokens, rope)
    patches = model.norm(tokens)[:, model.n_storage_tokens+1:]
    height, width = hw
    assert patches.shape[1] == height*width
    maps = patches.reshape(patches.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
    return F.adaptive_avg_pool2d(maps, (2, 2)).flatten(1)


def full_features(model, images):
    maps = model.get_intermediate_layers(images, n=1, reshape=True, norm=True)[-1]
    return F.adaptive_avg_pool2d(maps, (2, 2)).flatten(1)


class DirectionHead(nn.Module):
    def __init__(self, dim=5120):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim+1, 3)

    def forward(self, features, available=None):
        features = features.float()
        if available is None:
            available = torch.ones(len(features), 1, device=features.device, dtype=features.dtype)
        else:
            available = available.to(features).reshape(-1, 1)
        values = self.norm(features)*available
        return self.linear(torch.cat([values, available], dim=1))
