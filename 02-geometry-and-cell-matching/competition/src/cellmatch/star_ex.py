"""Fixed-budget EX star-convex regression initialized from a held-out ResNet.

Predict normalized center probability and 32 radial boundary distances. This is
an experimental independent implementation, not a pretrained StarDist model.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .resnet_ex import ResNetEX
from .resnet_crossfit import load_subset
from .learning import target_maps, choose_device, image_features
from .star_geometry import RAYS, RAY_SCALE, star_targets, extract_candidates, render_candidates
from .cellpose_baseline import confidence_maps
from .data import read_rows, write_rows, read_image, image_path, split_id
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cli import evaluate


def make_model(initial=None):
    model = ResNetEX()
    if initial is not None:
        model.load_state_dict(initial['state_dict'])
    old = model.out
    model.out = nn.Conv2d(32, 1+RAYS, 1)
    with torch.no_grad():
        model.out.weight.zero_()
        model.out.bias.fill_(math.log(math.expm1(6./RAY_SCALE)))
        if initial is not None:
            model.out.weight[:1].copy_(old.weight[1:2])
            model.out.bias[:1].copy_(old.bias[1:2])
    return model


def sample(items, rng, batch=4, patch=96):
    subjects = sorted({r['subject'] for r in items})
    xs, ps, ds, vs = [], [], [], []
    for _ in range(batch):
        subject = subjects[int(rng.integers(len(subjects)))]
        group = [r for r in items if r['subject'] == subject]
        record = group[int(rng.integers(len(group)))]; h, w = record['labels'].shape
        if rng.random() < .75 and len(record['centers']):
            cy, cx = record['centers'][int(rng.integers(len(record['centers'])))] + rng.uniform(-patch/4, patch/4, 2)
        else:
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        y = int(np.clip(cy-patch/2, 0, max(h-patch, 0)))
        x = int(np.clip(cx-patch/2, 0, max(w-patch, 0)))
        feature = record['x'][:, y:y+patch, x:x+patch].astype(np.float32)
        labels = record['labels'][y:y+patch, x:x+patch]
        ph, pw = patch-labels.shape[0], patch-labels.shape[1]
        feature = np.pad(feature, ((0, 0), (0, ph), (0, pw)))
        labels = np.pad(labels, ((0, ph), (0, pw)))
        if rng.random() < .5:
            feature, labels = feature[:, :, ::-1].copy(), labels[:, ::-1].copy()
        matrix = cv2.getRotationMatrix2D((patch/2, patch/2), rng.uniform(0, 360), rng.uniform(.8, 1.25))
        feature = np.stack([cv2.warpAffine(f, matrix, (patch, patch), flags=cv2.INTER_LINEAR) for f in feature])
        labels = cv2.warpAffine(labels.astype(np.float32), matrix, (patch, patch), flags=cv2.INTER_NEAREST).astype(np.int32)
        probability, distances, valid = star_targets(labels)
        feature = np.clip(feature**rng.uniform(.7, 1.4)*rng.uniform(.8, 1.2)+rng.normal(0, .01, feature.shape), 0, 1).astype(np.float32)
        xs.append(np.concatenate([feature, np.ones((1, patch, patch), np.float32)]))
        ps.append(probability); ds.append(distances); vs.append(valid)
    return tuple(torch.from_numpy(np.stack(v)) for v in (xs, ps, ds, vs))


def star_loss(logits, probability, distances, valid):
    weight = valid*(1+4*(probability > 0))
    obj = F.binary_cross_entropy_with_logits(logits[:, 0], probability, reduction='none')
    obj = (obj*weight).sum()/weight.sum().clamp_min(1)
    radial = F.softplus(logits[:, 1:])*RAY_SCALE
    weighted = probability[:, None]*valid[:, None]
    geometry = ((radial-distances).abs()*weighted).sum()/(weighted.sum()*RAYS).clamp_min(1)
    return obj+.2*geometry


@torch.inference_mode()
def tiled_prediction(model, image, device, tile=96, overlap=48, batch=4):
    features = image_features(image); h, w = image.shape
    def starts(size):
        return [0] if size <= tile else sorted(set(list(range(0, size-tile+1, tile-overlap))+[size-tile]))
    positions = [(y, x) for y in starts(h) for x in starts(w)]
    accumulated = np.zeros((1+RAYS, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), .01).astype(np.float32)
    for offset in range(0, len(positions), batch):
        subset = positions[offset:offset+batch]; xs = []
        for y, x in subset:
            crop = features[:, y:y+tile, x:x+tile]
            crop = np.pad(crop, ((0, 0), (0, tile-crop.shape[1]), (0, tile-crop.shape[2])), mode='reflect')
            xs.append(np.concatenate([crop, np.ones((1, tile, tile), np.float32)]))
        logits = model(torch.from_numpy(np.stack(xs)).to(device))
        prediction = torch.cat([logits[:, :1].sigmoid(), F.softplus(logits[:, 1:])*RAY_SCALE], 1).cpu().numpy()
        for (y, x), value in zip(subset, prediction):
            hh, ww = min(tile, h-y), min(tile, w-x)
            accumulated[:, y:y+hh, x:x+ww] += value[:, :hh, :ww]*window[:hh, :ww]
            weight[y:y+hh, x:x+ww] += window[:hh, :ww]
    return accumulated/weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage20_star_ex'))
    parser.add_argument('--initializer-root', type=Path, default=Path('outputs/stage16_resnet_ex'))
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--infer-development', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Positive training budget required')
    torch.set_num_threads(2); device = choose_device(args.device)
    root = Path('data/raw/Project_2_Dataset'); truth = read_rows(root/'training/train_ground_truth.csv')
    subject = FOLDS[args.fold]; ids = sorted(s for s in truth if split_id(s)[0] != subject)
    initializer = args.initializer_root/args.fold/'model.pt'
    initial = torch.load(initializer, map_location='cpu', weights_only=False)
    if initial['held_out'] != subject or set(initial['train_ids']) != set(ids):
        raise ValueError('Initializer contains evaluation mouse or wrong training partition')
    config = {'held_out': subject, 'train_ids': ids, 'steps': args.steps, 'seed': 2067,
              'rays': RAYS, 'ray_scale': RAY_SCALE, 'batch': 4, 'patch': 96,
              'sampling': '75% annotated-cell-centered, 25% random; balanced development mice',
              'encoder_lr': .00005, 'decoder_lr': .0003, 'head_lr': .001, 'ema_decay': .99,
              'initializer': str(initializer), 'initializer_sha256': hashlib.sha256(initializer.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'geometry_sha256': hashlib.sha256(Path(__file__).with_name('star_geometry.py').read_bytes()).hexdigest(),
              'loader_sha256': hashlib.sha256(Path(__file__).with_name('resnet_crossfit.py').read_bytes()).hexdigest(),
              'learning_sha256': hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest(),
              'encoder_sha256': hashlib.sha256(Path(__file__).with_name('resnet_ex.py').read_bytes()).hexdigest()}
    folder = args.output/args.fold; folder.mkdir(parents=True, exist_ok=True)
    config_path = folder/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Changed star training configuration')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    torch.manual_seed(2067); rng = np.random.default_rng(2067)
    final, resume = folder/'model.pt', folder/'resume.pt'
    model = make_model(None if final.exists() or resume.exists() else initial).to(device)
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        if state['config'] != config or state['step'] != args.steps:
            raise ValueError('Invalid completed star model')
        model.load_state_dict(state['state_dict'])
    else:
        items = load_subset(root, truth, ids)
        for record in items:
            if record['subject'] == subject:
                raise ValueError('Evaluation mouse in star training')
            _, record['labels'] = target_maps(truth[record['sample_id']]['exvivo_instances'], record['y'].shape[1:])
        ema = copy.deepcopy(model).eval()
        encoder = list(model.encoder.parameters()); head = list(model.out.parameters())
        special = {id(p) for p in encoder+head}
        decoder = [p for p in model.parameters() if id(p) not in special]
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': .00005}, {'params': decoder, 'lr': .0003},
                                       {'params': head, 'lr': .001}], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.000005)
        start, history = 0, []
        if resume.exists():
            state = torch.load(resume, map_location=device, weights_only=False)
            if state['config'] != config:
                raise ValueError('Changed star resume configuration')
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            rng.bit_generator.state = state['rng']; start = state['step']; history = state['history']
        model.train(); started = time.perf_counter()
        for step in range(start+1, args.steps+1):
            x, p, d, v = (t.to(device) for t in sample(items, rng))
            optimizer.zero_grad(set_to_none=True); loss = star_loss(model(x), p, d, v); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); scheduler.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.mul_(.99).add_(current, alpha=.01)
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite star loss')
            if step == 1 or step % 100 == 0:
                entry = {'step': step, 'loss': value, 'seconds': time.perf_counter()-started}
                history.append(entry); print('Star EX', args.fold, entry, flush=True)
            if step % 500 == 0 or step == args.steps:
                temporary = folder/'resume.tmp'
                torch.save({'config': config, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'rng': rng.bit_generator.state, 'history': history}, temporary)
                temporary.replace(resume)
        torch.save({'config': config, 'step': args.steps, 'held_out': subject, 'train_ids': ids,
                    'architecture': 'resnet18_star32', 'state_dict': {k: v.detach().cpu() for k, v in ema.state_dict().items()}}, final)
        (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        model = ema; del items, optimizer, encoder, decoder, head, loss, x, p, d, v
    model.eval()
    cache = folder/'candidates'; cache.mkdir(exist_ok=True)
    maps_folder = folder/'maps'; maps_folder.mkdir(exist_ok=True)
    reference = read_rows('outputs/stage13_native_relative/preserved/predicted_all.csv'); rows = []
    partial = folder/'regions'; partial.mkdir(exist_ok=True)
    for sid in truth:
        heldout = split_id(sid)[0] == subject
        if not heldout and not args.infer_development:
            continue
        path = cache/f'{sid}.npz'
        if not path.exists():
            image = read_image(image_path(root, 'training', sid, 'exvivo'))
            prediction = tiled_prediction(model, image, device)
            points, radii, scores = extract_candidates(prediction[0], prediction[1:])
            np.savez_compressed(path, probability=prediction[0].astype(np.float16), points=points,
                                radii=radii.astype(np.float16), scores=scores.astype(np.float16))
            del prediction
        with np.load(path) as values:
            probability = values['probability'].astype(np.float32)
            labels = render_candidates(values['points'], values['radii'].astype(np.float32), values['scores'].astype(np.float32), probability.shape)
        clipped = np.clip(probability, 1e-6, 1-1e-6)
        maps = confidence_maps(labels, np.log(clipped/(1-clipped)))
        np.savez_compressed(maps_folder/f'{sid}_exvivo.npz', maps=maps.astype(np.float16))
        row = copy.deepcopy(reference[sid]); row['match_pairs'] = []
        row['exvivo_instances'] = labels_to_instances(labels, 'STAR')
        (partial/f'{sid}.json').write_text(json.dumps(row)+'\n')
        if heldout:
            rows.append(row)
        print('Star EX inference', sid, len(row['exvivo_instances']), flush=True)
    write_rows(rows, folder/'heldout.csv')
    evaluate(SimpleNamespace(data=root, predictions=folder/'heldout.csv', subject=subject, output=folder/'metrics.json'))


if __name__ == '__main__':
    main()
