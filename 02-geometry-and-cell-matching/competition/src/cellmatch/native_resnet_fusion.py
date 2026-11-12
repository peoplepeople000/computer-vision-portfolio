"""Trainable public image features injected into the original native contour net."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .learning import SmallUNet


class NativeResNetFusion(SmallUNet):
    def __init__(self, pretrained=None):
        super().__init__(16)
        # Preserve the native initializer and all subsequent CPU RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.encoder = resnet18(weights=None)
            if pretrained is not None:
                self.encoder.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
            self.encoder.fc = nn.Identity()
            self.inject_b = nn.Conv2d(64, 32, 1, bias=False)
            self.inject_c = nn.Conv2d(128, 64, 1, bias=False)
            self.inject_context = nn.Conv2d(256 + 512, 64, 1, bias=False)
            for layer in (self.inject_b, self.inject_c, self.inject_context):
                nn.init.zeros_(layer.weight)
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Retain ImageNet running statistics; affine BN parameters remain trainable.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == (3, 32, 32)
        rgb = F.interpolate(x[:, :1].repeat(1, 3, 1, 1), size=(64, 64), mode='bilinear', align_corners=False)
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        enc = self.encoder
        stem = enc.relu(enc.bn1(enc.conv1(rgb)))
        f1 = enc.layer1(enc.maxpool(stem))
        f2 = enc.layer2(f1)
        f3 = enc.layer3(f2)
        f4 = enc.layer4(f3)
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2)) + self.inject_b(f1)
        context = torch.cat([F.interpolate(f, size=(8, 8), mode='bilinear', align_corners=False) for f in (f3, f4)], 1)
        c = self.c(F.avg_pool2d(b, 2)) + self.inject_c(f2) + self.inject_context(context)
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        return self.out(self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1)))


def optimizer(model):
    native = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
    return torch.optim.AdamW([{'params': native, 'lr': .001}, {'params': model.encoder.parameters(), 'lr': .00001}], weight_decay=.0001)


def scheduler(opt):
    # Separate minima: 1e-4 native, 1e-6 encoder. A shared absolute minimum
    # would incorrectly increase the encoder's learning rate.
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: .1 + .9 * (1 + math.cos(math.pi * step / 3000)) / 2)


def average_states(snapshots):
    result = {}
    for key, value in snapshots[0].items():
        frozen_buffer = key in ('rgb_mean', 'rgb_std') or any(key.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked'))
        if value.is_floating_point() and not frozen_buffer:
            result[key] = torch.stack([state[key] for state in snapshots]).mean(0)
        else:
            assert all(torch.equal(value, state[key]) for state in snapshots)
            result[key] = value.clone()
    return result


def bn_buffers(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith('encoder.') and any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked'))}
