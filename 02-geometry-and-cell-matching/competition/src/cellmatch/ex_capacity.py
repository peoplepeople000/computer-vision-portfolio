"""Larger EX-only model with training-only hard-negative mining and native tiles.

Every fold uses a fixed training budget. All calibration images belong to its
two training subjects; the third subject is only evaluated after fitting.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage as ndi

from .learning import SmallUNet, load_training, choose_device, probability_maps, instances_from_maps
from .ex_specialist import ex_loss
from .data import read_rows, write_rows, read_image, image_path, parse_instances, split_id
from .rle import labels_to_instances
from .ex_joint_calibration import morphology_features
from .stage3 import FOLDS, calibrate_quality, quality_probabilities
from .metrics import segmentation_score
from .cli import evaluate


def mine_centers(foreground, maps, centers):
    # Ignore candidates within five pixels of an annotated cell to avoid
    # treating an imperfect boundary or off-center prediction as background.
    maps = np.asarray(maps, dtype=np.float32)
    far = ndi.distance_transform_edt(~foreground.astype(bool)) > 5
    peaks = (maps[0] > .5) & (maps[0] == ndi.maximum_filter(maps[0], size=9)) & far
    negatives = np.argwhere(peaks).astype(float)
    index = np.rint(centers).astype(int)
    hard = centers[maps[0, index[:, 0], index[:, 1]] < .7] if len(centers) else centers
    return hard, negatives


def sample(items, rng, batch=8, patch=128):
    subjects = sorted({r['subject'] for r in items})
    xs, ys = [], []
    for _ in range(batch):
        subject = subjects[int(rng.integers(len(subjects)))]
        candidates = [r for r in items if r['subject'] == subject]
        r = candidates[int(rng.integers(len(candidates)))]
        h, w = r['x'].shape[1:]
        choice = rng.random()
        centers = r['hard'] if choice < .2 else r['negative'] if choice < .35 else r['centers']
        if choice < .85 and len(centers):
            cy, cx = centers[int(rng.integers(len(centers)))] + rng.uniform(-patch/3, patch/3, 2)
        else:
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        y = int(np.clip(cy-patch/2, 0, max(0, h-patch)))
        x = int(np.clip(cx-patch/2, 0, max(0, w-patch)))
        feature = r['x'][:, y:y+patch, x:x+patch].astype(np.float32)
        target = r['y'][:, y:y+patch, x:x+patch].astype(np.float32)
        pad = ((0, 0), (0, patch-feature.shape[1]), (0, patch-feature.shape[2]))
        feature, target = np.pad(feature, pad, mode='reflect'), np.pad(target, pad)
        k = int(rng.integers(4))
        feature, target = np.rot90(feature, k, (1, 2)), np.rot90(target, k, (1, 2))
        if rng.random() < .5:
            feature, target = feature[:, :, ::-1], target[:, :, ::-1]
        if rng.random() < .2:
            feature = ndi.gaussian_filter(feature, (0, *([rng.uniform(.3, 1.)]*2)))
        feature = np.clip(feature**rng.uniform(.5, 1.8)*rng.uniform(.65, 1.4)
                          + rng.normal(0, rng.uniform(0, .025), feature.shape), 0, 1).astype(np.float32)
        xs.append(np.concatenate([feature, np.ones((1, patch, patch), np.float32)]))
        ys.append(target.copy())
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def train_fold(folder, tag, args, truth):
    heldout = FOLDS[tag]
    ids = sorted(s for s in truth if split_id(s)[0] != heldout)
    config = {'held_out': heldout, 'train_ids': ids, 'width': 24, 'steps': args.steps, 'seed': 2040,
              'batch': 8, 'patch': 128, 'ema_decay': .995,
              'sampling': '20% difficult positive, 15% hard background, 50% positive, 15% random; balanced subjects',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'learning_sha256': hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest()}
    config_path = folder/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Changed training configuration')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    device = choose_device(args.device)
    model = SmallUNet(24).to(device)
    checkpoint = folder/'model.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['config'] != config or state['step'] != args.steps:
            raise ValueError('Incomplete or incompatible final checkpoint')
        model.load_state_dict(state['state_dict'])
        return model.eval(), device, config
    items, loaded = load_training(Path(args.data), heldout)
    items = [r for r in items if r['mode'] == 1]
    if set(loaded) != set(ids) or any(r['subject'] == heldout for r in items):
        raise ValueError('Invalid training partition')
    dev = Path(f'outputs/stage6_ex/development_{tag}')
    provenance = json.loads((dev/'manifest.json').read_text())
    if provenance['held_out'] != heldout or set(provenance['train_ids']) != set(ids):
        raise ValueError('Mining model contains evaluation data')
    for r in items:
        maps = np.load(dev/f"{r['sample_id']}_exvivo.npz")['maps']
        r['hard'], r['negative'] = mine_centers(r['y'][0], maps, r['centers'])
    torch.manual_seed(2040)
    # Initialize after fixing the seed, including on resumed runs.
    model = SmallUNet(24).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00005)
    rng = np.random.default_rng(2040)
    start_step, history = 0, []
    resume = folder/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['rng']; start_step = state['step']; history = state['history']
    started = time.perf_counter()
    model.train()
    for step in range(start_step+1, args.steps+1):
        x, y = sample(items, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = ex_loss(model(x.to(device)), y.to(device)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.mul_(.995).add_(current, alpha=.005)
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite training loss')
        if step == 1 or step % 200 == 0:
            entry = {'step': step, 'loss': value, 'run_seconds': time.perf_counter()-started}
            history.append(entry); print('EX capacity', tag, entry, flush=True)
        if step % 500 == 0 or step == args.steps:
            temporary = folder/'resume.tmp'
            torch.save({'config': config, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng': rng.bit_generator.state, 'history': history}, temporary)
            temporary.replace(resume)
    torch.save({'config': config, 'step': args.steps, 'width': 24, 'held_out': heldout, 'train_ids': ids,
                'state_dict': {k: v.detach().cpu() for k, v in ema.state_dict().items()}}, checkpoint)
    (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    return ema.eval(), device, config


def calibrate(folder, tag, truth, reference):
    subject = FOLDS[tag]
    ids = sorted(s for s in truth if split_id(s)[0] != subject)
    config = json.loads((folder/'training_config.json').read_text())
    if config['train_ids'] != ids:
        raise ValueError('Invalid calibration partition')
    options = []
    for threshold in (.5, .6, .7):
        records = []
        for sid in ids:
            maps = np.load(folder/'maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            encoded = labels_to_instances(instances_from_maps(maps, threshold, .35), 'EXP')
            masks = parse_instances({'exvivo_instances': encoded}, 'exvivo', maps.shape[1:])
            keys, features = morphology_features(masks, maps)
            actual = parse_instances(truth[sid], 'exvivo', maps.shape[1:])
            mapping = segmentation_score(masks, actual)['pred_to_gt']
            records.append({'sample_id': sid, 'subject': split_id(sid)[0], 'features': [None, features],
                            'quality': np.array([k in mapping for k in keys]), 'gt_count': len(actual),
                            'ious': np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0. for k in keys])})
        fitted, quality_threshold, measurements = calibrate_quality(records)
        score = next(r['mean_pq'] for r in measurements if r['threshold'] == quality_threshold)
        options.append((score, threshold, quality_threshold, fitted, measurements))
        print('EX calibration', tag, threshold, quality_threshold, score, flush=True)
    score, threshold, quality_threshold, fitted, measurements = max(options, key=lambda r: r[0])
    torch.save({'quality': fitted[0].state_dict(), 'mean': fitted[1], 'sd': fitted[2],
                'foreground': threshold, 'seed_threshold': .35, 'threshold': quality_threshold,
                'train_ids': ids, 'held_out': subject}, folder/'quality.pt')
    (folder/'calibration.json').write_text(json.dumps({
        'train_ids': ids, 'held_out': subject, 'foreground': threshold, 'quality': quality_threshold,
        'options': [{'foreground': r[1], 'quality': r[2], 'inner_pq': r[0], 'threshold_scores': r[4]} for r in options],
        'limitation': 'Inner quality classifier holdout; segmentation sees both development subjects. Outer subject excluded throughout.'}, indent=2)+'\n')
    results = []
    for sid in truth:
        if split_id(sid)[0] != subject:
            continue
        maps = np.load(folder/'maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
        encoded = labels_to_instances(instances_from_maps(maps, threshold, .35), 'EXP')
        masks = parse_instances({'exvivo_instances': encoded}, 'exvivo', maps.shape[1:])
        keys, features = morphology_features(masks, maps)
        probabilities = quality_probabilities(fitted, features)
        keep = {key for key, p in zip(keys, probabilities) if p >= quality_threshold}
        value = copy.deepcopy(reference[sid]); value['match_pairs'] = []
        value['exvivo_instances'] = {k: v for k, v in encoded.items() if k in keep}
        results.append(value)
    write_rows(results, folder/'heldout.csv')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw/Project_2_Dataset')
    parser.add_argument('--output', default='outputs/stage9_ex_capacity')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--fold', choices=['all', *FOLDS], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.data); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    truth = read_rows(root/'training/train_ground_truth.csv')
    reference = read_rows('outputs/stage8_iv_quality_capacity/joint/predicted_all.csv')
    for tag in (FOLDS if args.fold == 'all' else [args.fold]):
        folder = out/tag; folder.mkdir(exist_ok=True)
        if (folder/'heldout.csv').exists():
            continue
        model, device, config = train_fold(folder, tag, args, truth)
        cache = folder/'maps'; cache.mkdir(exist_ok=True)
        for sid in truth:
            path = cache/f'{sid}_exvivo.npz'
            if path.exists():
                continue
            image = read_image(image_path(root, 'training', sid, 'exvivo'))
            maps = probability_maps(model, image, 1, device, tile=128, overlap=64, batch_size=8)
            np.savez_compressed(path, maps=maps.astype(np.float16))
            print('EX native maps', tag, sid, flush=True)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
        calibrate(folder, tag, truth, reference)
        evaluate(SimpleNamespace(data=root, predictions=folder/'heldout.csv', subject=FOLDS[tag], output=folder/'metrics.json'))
    if args.fold == 'all':
        rows = [r for tag in FOLDS for r in read_rows(out/tag/'heldout.csv').values()]
        write_rows(rows, out/'segmentation_oof.csv')
        evaluate(SimpleNamespace(data=root, predictions=out/'segmentation_oof.csv', subject=None, output=out/'segmentation_oof_metrics.json'))


if __name__ == '__main__':
    main()
