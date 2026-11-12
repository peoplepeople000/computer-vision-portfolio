"""Pretrained high-resolution EX U-Net with fixed-budget outer-mouse training.

Use .venv-cellpose for its existing torchvision installation. Public ImageNet
weights initialize the encoder; competition training excludes the entire fold.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning import Block, load_training, sample_batch, choose_device, probability_maps, instances_from_maps
from .ex_specialist import ex_loss
from .data import read_rows, read_image, image_path, split_id, write_rows
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cli import evaluate

WEIGHTS = Path('models/torchvision/checkpoints/resnet18-f37072fd.pth')


class ResNetEX(nn.Module):
    def __init__(self, initializer=None):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder = resnet18(weights=None)
        if initializer is not None:
            self.encoder.load_state_dict(torch.load(initializer, map_location='cpu', weights_only=True))
        self.encoder.fc = nn.Identity()
        self.d3 = Block(512+256, 128)
        self.d2 = Block(128+128, 96)
        self.d1 = Block(96+64, 64)
        self.d0 = Block(64+64, 32)
        self.out = nn.Conv2d(32, 2, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained running statistics fixed for small microscopy batches.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        # Twofold input enlargement keeps stem features at native pixel resolution.
        rgb = torch.cat([x[:, :1], x[:, :1], x[:, 1:2]], 1)
        rgb = F.interpolate(rgb, scale_factor=2., mode='bilinear', align_corners=False)
        rgb = (rgb-self.mean)/self.std
        e = self.encoder
        a = e.relu(e.bn1(e.conv1(rgb)))
        b = e.layer1(e.maxpool(a)); c = e.layer2(b)
        d = e.layer3(c); f = e.layer4(d)
        def up(z, skip, block):
            return block(torch.cat([F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        z = up(f, d, self.d3); z = up(z, c, self.d2)
        z = up(z, b, self.d1); z = up(z, a, self.d0)
        return F.interpolate(self.out(z), size=x.shape[-2:], mode='bilinear', align_corners=False)


def spatial_augment(x, y, rng):
    """Apply the same continuous rotation/scale to images and soft targets."""
    transforms = []
    for _ in range(len(x)):
        angle = rng.uniform(-math.pi, math.pi)
        scale = rng.uniform(.8, 1.25)
        transforms.append([[math.cos(angle)/scale, -math.sin(angle)/scale, 0],
                           [math.sin(angle)/scale, math.cos(angle)/scale, 0]])
    theta = torch.tensor(transforms, dtype=x.dtype)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    y = F.grid_sample(y, grid, align_corners=False, padding_mode='zeros')
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage16_resnet_ex'))
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--infer-development', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    root = Path('data/raw/Project_2_Dataset'); subject = FOLDS[args.fold]
    truth = read_rows(root/'training/train_ground_truth.csv')
    ids = sorted(s for s in truth if split_id(s)[0] != subject)
    folder = args.output/args.fold; folder.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        raise FileNotFoundError('Download official resnet18-f37072fd.pth into models/torchvision/checkpoints first')
    config = {'architecture': 'ResNet18 encoder + native-resolution U-Net decoder',
              'held_out': subject, 'train_ids': ids, 'seed': 2061, 'steps': args.steps,
              'patch': 96, 'batch': 4, 'encoder_lr': .0001, 'decoder_lr': .001,
              'ema_decay': .99, 'normalization': 'Frozen encoder BatchNorm + decoder GroupNorm',
              'torch_version': torch.__version__, 'torchvision_version': version('torchvision'),
              'augmentation': 'Subject-balanced 70% cell-centered sampling, rotations, scale .8..1.25, gamma/gain/noise',
              'initializer': str(WEIGHTS), 'initializer_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'learning_sha256': hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest()}
    config_path = folder/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Training configuration changed; preserve earlier run')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    device = choose_device(args.device)
    torch.manual_seed(2061); rng = np.random.default_rng(2061)
    final = folder/'model.pt'; resume = folder/'resume.pt'
    model = ResNetEX(None if final.exists() or resume.exists() else WEIGHTS).to(device)
    if final.exists():
        saved = torch.load(final, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['step'] != args.steps:
            raise ValueError('Invalid final checkpoint')
        model.load_state_dict(saved['state_dict'])
    else:
        items, loaded = load_training(root, subject)
        items = [r for r in items if r['mode'] == 1]
        if set(loaded) != set(ids) or any(r['subject'] == subject for r in items):
            raise ValueError('Invalid training partition')
        ema = copy.deepcopy(model).eval()
        encoder = list(model.encoder.parameters()); encoder_ids = {id(p) for p in encoder}
        decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': .0001}, {'params': decoder, 'lr': .001}], weight_decay=1e-4)
        def multiplier(step):
            return min((step+1)/100., 1.)*(.05+.95*(1+math.cos(math.pi*step/args.steps))/2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        start, history = 0, []
        if resume.exists():
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved['config'] != config:
                raise ValueError('Resume configuration changed')
            model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            rng.bit_generator.state = saved['rng']; start = saved['step']; history = saved['history']
        model.train(); started = time.perf_counter()
        for step in range(start+1, args.steps+1):
            x, y = sample_batch(items, 4, 96, rng)
            x, y = spatial_augment(x, y, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = ex_loss(model(x.to(device)), y.to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); scheduler.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.mul_(.99).add_(current, alpha=.01)
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite training loss')
            if step == 1 or step % 100 == 0:
                entry = {'step': step, 'loss': value, 'seconds': time.perf_counter()-started}
                history.append(entry); print('ResNet EX', args.fold, entry, flush=True)
            if step % 500 == 0 or step == args.steps:
                tmp = folder/'resume.tmp'
                torch.save({'config': config, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'rng': rng.bit_generator.state, 'history': history}, tmp)
                tmp.replace(resume)
        torch.save({'config': config, 'step': args.steps, 'held_out': subject, 'train_ids': ids,
                    'architecture': 'resnet18_ex', 'state_dict': {k: v.detach().cpu() for k, v in ema.state_dict().items()}}, final)
        (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        model = ema; del items, optimizer, encoder, decoder, loss, x, y
    model.eval()
    maps_folder = folder/'maps'; maps_folder.mkdir(exist_ok=True)
    reference = read_rows('outputs/stage13_native_relative/preserved/predicted_all.csv')
    rows = []
    for sid in truth:
        heldout = split_id(sid)[0] == subject
        if not heldout and not args.infer_development:
            continue
        cached = maps_folder/f'{sid}_exvivo.npz'
        if cached.exists():
            maps = np.load(cached)['maps'].astype(np.float32)
        else:
            image = read_image(image_path(root, 'training', sid, 'exvivo'))
            maps = probability_maps(model, image, 1, device, tile=96, overlap=48, batch_size=4)
            np.savez_compressed(cached, maps=maps.astype(np.float16))
            # Always use persisted precision so resumed and fresh predictions agree.
            maps = maps.astype(np.float16).astype(np.float32)
        if heldout:
            row = copy.deepcopy(reference[sid]); row['match_pairs'] = []
            row['exvivo_instances'] = labels_to_instances(instances_from_maps(maps, .5, .35), 'EXP')
            rows.append(row)
        print('ResNet EX inference', sid, flush=True)
    write_rows(rows, folder/'heldout.csv')
    evaluate(SimpleNamespace(data=root, predictions=folder/'heldout.csv', subject=subject, output=folder/'metrics.json'))


if __name__ == '__main__':
    main()
