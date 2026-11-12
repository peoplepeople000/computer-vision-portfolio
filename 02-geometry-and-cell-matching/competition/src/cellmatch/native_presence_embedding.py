"""Frozen public ResNet18 mask/context features for EX annotation presence."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .ex_boundary import origin, local_mask


def crops(features, masks, keys):
    result = []
    for k in keys:
        x, y = origin(masks[k], 48)
        prior = local_mask(masks[k], x, y, 48)
        chip = np.concatenate([features[:, y:y+48, x:x+48], prior[None]])
        assert chip.shape == (3, 48, 48) and prior.any()
        result.append(chip)
    return np.stack(result).astype(np.float16)


class PublicPresenceEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('sd', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x):
        rgb = torch.cat([x[:, :1], x[:, :1], x[:, 1:2]], 1)
        rgb = F.interpolate(rgb, scale_factor=2., mode='bilinear', align_corners=False)
        rgb = (rgb-self.mean)/self.sd
        e = self.encoder
        z = e.relu(e.bn1(e.conv1(rgb)))
        z = e.maxpool(z)
        vectors = []
        for block in [e.layer1, e.layer2, e.layer3]:
            z = block(z)
            mask = F.interpolate(x[:, 2:3], size=z.shape[-2:], mode='area')
            for weights in [mask, 1-mask]:
                vectors.append((z*weights).sum((2, 3))/weights.sum((2, 3)).clamp_min(1e-6))
        return torch.cat(vectors, 1)


def embed_d4(model, chips, device, batch=64):
    output = []
    with torch.inference_mode():
        for start in range(0, len(chips), batch):
            x = torch.from_numpy(chips[start:start+batch].astype(np.float32)).to(device)
            values = []
            for flip in [False, True]:
                for turns in range(4):
                    view = torch.rot90(x, turns, (2, 3))
                    if flip:
                        view = view.flip(3)
                    values.append(model(view).cpu().numpy())
            output.append(np.mean(values, axis=0))
    result = np.concatenate(output)
    assert result.shape == (len(chips), 896) and np.isfinite(result).all()
    return result.astype(np.float32)
