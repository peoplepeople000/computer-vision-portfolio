"""Frozen Cellpose spatial fields and a zero-initialized native feature branch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cellpose.resnet_torch import CPnet
from .learning import SmallUNet


class CytoSpatialEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.encoder = CPnet([2, 32, 64, 128, 256], 3, 3, mkldnn=False)
        self.encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        image = torch.cat([x[:, :1], torch.zeros_like(x[:, :1])], 1)
        image = F.interpolate(image, scale_factor=3., mode='bilinear', align_corners=False)
        features = self.encoder.downsample(image)
        style = self.encoder.make_style(features[-1])
        spatial = self.encoder.upsample(style, features, False)
        return F.interpolate(spatial, size=x.shape[-2:], mode='bilinear', align_corners=False)


@torch.inference_mode()
def spatial_d4(encoder, x):
    fields = []
    for flip in [False, True]:
        base = x.flip(-1) if flip else x
        for turns in range(4):
            field = encoder(torch.rot90(base, turns, (-2, -1)))
            field = torch.rot90(field, -turns, (-2, -1))
            if flip: field = field.flip(-1)
            fields.append(field)
    return torch.stack(fields).mean(0)


def extract_fields(encoder, chips, device, batch=32):
    outputs = []
    for start in range(0, len(chips), batch):
        x = torch.from_numpy(chips[start:start+batch].astype(np.float32)).to(device)
        outputs.append(spatial_d4(encoder, x).cpu().numpy())
    fields = np.concatenate(outputs).astype(np.float32)
    assert fields.shape == (len(chips), 32, 32, 32) and np.isfinite(fields).all()
    return fields


class FeatureUNet(SmallUNet):
    def __init__(self, mean=None, sd=None):
        super().__init__(16)
        old = self.a[0]
        state = torch.get_rng_state()
        expanded = nn.Conv2d(35, 16, 3, padding=1)
        torch.set_rng_state(state)
        with torch.no_grad():
            expanded.weight.zero_(); expanded.weight[:, :3].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.a[0] = expanded
        self.register_buffer('feature_mean', torch.as_tensor(np.zeros(32) if mean is None else mean, dtype=torch.float32).reshape(1, 32, 1, 1))
        self.register_buffer('feature_sd', torch.as_tensor(np.ones(32) if sd is None else sd, dtype=torch.float32).reshape(1, 32, 1, 1))

    def forward(self, x):
        assert x.shape[1] == 35
        field = (x[:, 3:]-self.feature_mean)/self.feature_sd
        return super().forward(torch.cat([x[:, :3], field], 1))
