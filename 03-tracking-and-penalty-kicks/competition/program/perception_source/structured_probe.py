"""R2: cached 2D goal/ball/role geometry, independent of RGB and pose.

Goal boxes are not calibrated posts. Coordinate differences are not tracked
physical velocities. All supplied anchors precede the dataset clip boundary.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import rgb_gru as model
import rgb_gru_data as data
import research_loop as research
import sparse_role_comparison as sparse

VARIANTS = ('quality_only', 'absolute_last', 'relative_last', 'relative_motion')
CONFIGS = ({'kind': 'linear', 'decay': .01}, {'kind': 'mlp', 'decay': .01})


def bottom(box):
    return np.array([(box[0] + box[2]) / 2, box[3]], dtype=np.float32)


def anchor_features(anchor, selected):
    """Return scale/translation-invariant coordinates and explicit missingness."""
    choice = selected['selection']
    if choice is None:
        return np.zeros(16, np.float32), np.zeros(16, np.float32), np.zeros(4, np.float32)
    goal, ball = [np.asarray(choice[k]['bbox'], np.float32) for k in ('goal', 'ball')]
    boxes = [np.asarray(selected['roles'][k], np.float32) for k in ('kicker', 'keeper')]
    if any(b.shape != (4,) or not np.isfinite(b).all() for b in [goal, ball, *boxes]):
        raise ValueError('Invalid selected detection')
    wh = goal[2:] - goal[:2]
    if (wh <= 1).any():
        return np.zeros(16, np.float32), np.zeros(16, np.float32), np.zeros(4, np.float32)
    scale = np.array([anchor['width'], anchor['height']], np.float32)
    absolute = np.concatenate([b / np.tile(scale, 2) for b in [goal, ball, *boxes]])
    g, b = bottom(goal), bottom(ball)
    relative = np.concatenate([(b - g) / wh,
                               *[(bottom(p) - b) / wh for p in boxes],
                               *[(bottom(p) - g) / wh for p in boxes],
                               *[(p[2:] - p[:2]) / wh for p in boxes],
                               (ball[2:] - ball[:2]) / wh])
    quality = np.array([1., choice['goal']['score'], choice['ball']['score'], choice['score']], np.float32)
    return absolute, np.clip(relative, -10, 10), quality


def clip_features(detection):
    anchors, selected = detection['anchors'], detection['selected']
    assert len(anchors) == len(selected) == 3
    values = [anchor_features(a, s) for a, s in zip(anchors, selected)]
    absolute, relative, quality = map(np.stack, zip(*values))
    times = np.array([a['timestamp'] for a in anchors])
    assert (np.diff(times) > 0).all()
    pair = quality[1:, 0] * quality[:-1, 0]
    # Camera/viewpoint and independently selected identities can affect this.
    differences = np.diff(relative, axis=0) / np.diff(times)[:, None]
    differences = np.clip(differences, -20, 20) * pair[:, None]
    q = quality.flatten()
    return {'quality_only': q,
            'absolute_last': np.r_[absolute[-1], q],
            'relative_last': np.r_[relative[-1], q],
            'relative_motion': np.r_[relative.flatten(), differences.flatten(), pair, q]}


def pack(cache, folds, out):
    started = time.monotonic()
    rows = pd.read_csv(folds)
    assert len(rows) == rows['clip'].nunique() == 1299
    assert rows.groupby('source_id').fold.nunique().max() == 1
    assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    manifest = json.loads((cache / 'artifact_manifest.json').read_text())
    values, hashes = [], {}
    for clip in rows['clip']:
        name = f'detections/{clip}.json'
        digest = data.sha(cache / name)
        assert digest == manifest[name]
        hashes[name] = digest
        values.append(clip_features(json.loads((cache / name).read_text())))
    out.mkdir(parents=True, exist_ok=False)
    arrays = {v: np.stack([x[v] for x in values]).astype(np.float32) for v in VARIANTS}
    assert all(np.isfinite(x).all() for x in arrays.values())
    np.savez_compressed(out / 'geometry.npz', clips=rows['clip'].to_numpy(dtype=str), **arrays)
    rows.to_csv(out / 'public_folds.csv', index=False)
    report = {'clips': len(rows), 'dimensions': {k: v.shape[1] for k, v in arrays.items()},
              'seconds': time.monotonic() - started,
              'bytes': (out / 'geometry.npz').stat().st_size,
              'valid_last_pair_fraction': float((arrays['quality_only'][:, 8] > 0).mean()),
              'source_sha256': data.sha(__file__), 'detection_sha256': hashes,
              'original_folds_sha256': data.sha(folds),
              'files': {n: data.sha(out / n) for n in ['geometry.npz', 'public_folds.csv']},
              'limitations': 'Goal bounding box, not posts/homography. Availability is not role accuracy.'}
    data.dump(out / 'manifest.json', report)
    return {k: v for k, v in report.items() if k != 'detection_sha256'}


def normalized(x, indices):
    mean = x[indices].mean(0)
    scale = x[indices].std(0, unbiased=False).clamp_min(.1)
    return ((x - mean) / scale).clamp(-10, 10), mean, scale


def head_for(dim, config):
    if config['kind'] == 'linear':
        return nn.Linear(dim, 8)
    return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(.35), nn.Linear(64, 8))


def fit(x, meta, labels, indices, evaluation, config, epochs, out, deadline, seed):
    start = time.monotonic(); model.seed_all(seed)
    z, mean, scale = normalized(x, indices)
    features = torch.cat([z, meta], 1).cuda()
    offsets = research.foot_priors(meta, labels, indices).cuda()
    target, selected = labels.cuda(), indices.cuda()
    net = head_for(features.shape[1], config).cuda()
    final = net if config['kind'] == 'linear' else net[-1]
    nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=config['decay'])
    best_score, best_epoch, history = -1., 0, []
    for epoch in range(1, epochs + 1):
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded GPU run stopped; completed fits remain saved')
        net.train(); optimizer.zero_grad()
        logits = net(features[selected]) + offsets[selected]
        loss = sum(w * nn.functional.cross_entropy(a, target[selected, k])
                   for k, (w, a) in enumerate(zip(data.WEIGHTS, logits.split((3, 3, 2), 1))))
        assert torch.isfinite(loss); loss.backward(); optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            net.eval()
            with torch.no_grad(): logits = net(features) + offsets
            row = {'epoch': epoch, 'loss': float(loss.detach()),
                   'train_score': research.predict_metrics(logits, target, selected)[0]['weighted_score']}
            if evaluation is not None:
                row['inner_score'] = research.predict_metrics(logits, target, evaluation.cuda())[0]['weighted_score']
            history.append(row)
            score = row.get('inner_score', float(epoch))
            if score > best_score:
                best_score, best_epoch = score, epoch
                model.atomic_save({'model': {k: v.cpu() for k, v in net.state_dict().items()},
                                   'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                                   'cuda_rng': torch.cuda.get_rng_state_all(), 'epoch': epoch,
                                   'mean': mean, 'scale': scale, 'fit_indices': indices,
                                   'offsets': offsets.cpu(), 'config': config, 'seed': seed}, out / 'checkpoint.pt')
    data.dump(out / 'history.json', history)
    state = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True)
    net.cpu(); net.load_state_dict(state['model']); net.eval()
    with torch.no_grad(): logits = net(features.cpu()) + state['offsets']
    info = {'seconds': time.monotonic() - start, 'best_epoch': best_epoch,
            'inner_score': best_score if evaluation is not None else None,
            'parameters': sum(p.numel() for p in net.parameters()),
            'checkpoint_sha256': data.sha(out / 'checkpoint.pt'), 'config': config}
    print(json.dumps({'fit': str(out), **info}), flush=True)
    return logits, info


def execute(bundle, out, budget=600, seed=6643):
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4)
    start = time.monotonic(); deadline = start + budget - 60
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items(): assert data.sha(bundle / name) == digest
    assert manifest['source_sha256'] == data.sha(__file__)
    rows = pd.read_csv(bundle / 'public_folds.csv')
    arrays = np.load(bundle / 'geometry.npz', allow_pickle=False)
    np.testing.assert_array_equal(arrays['clips'], rows['clip'])
    meta = torch.nn.functional.one_hot(torch.tensor([{'left': 0, 'right': 1}.get(x, 2) for x in rows.preferred_foot]), 3).float()
    labels = torch.tensor([[classes.index(row[k]) for k, classes in data.TARGETS.items()] for _, row in rows.iterrows()])
    ix = lambda a: torch.tensor(np.flatnonzero(a))
    train, val, inner, check = ix(rows.fold != 0), ix(rows.fold == 0), ix(~rows.fold.isin([0, 1])), ix(rows.fold == 1)
    assert set(rows.iloc[train].source_id).isdisjoint(rows.iloc[val].source_id)
    assert set(rows.iloc[inner].source_id).isdisjoint(rows.iloc[check].source_id)
    protocol = {'source_sha256': data.sha(__file__), 'bundle_sha256': data.sha(bundle / 'manifest.json'),
                'configs': CONFIGS, 'variants': VARIANTS, 'seed': seed, 'inner_fold': 1, 'outer_fold': 0,
                'max_epochs': 200, 'learning_rate': .003, 'gpu': torch.cuda.get_device_name(0),
                'helper_sha256': {Path(m.__file__).name: data.sha(m.__file__) for m in [model, data, research, sparse]}}
    data.immutable_json(out / 'protocol.json', json.loads(json.dumps(protocol)))
    baseline, base_pred, _ = research.predict_metrics(research.foot_priors(meta, labels, train), labels, val)
    results = {'foot_rule': {'metrics': baseline}}
    try:
        for variant in VARIANTS:
            directory = out / variant
            x = torch.from_numpy(arrays[variant]); candidates = []
            for config in CONFIGS:
                _, info = fit(x, meta, labels, inner, check, config, 200,
                              directory / ('inner_' + config['kind']), deadline, seed)
                candidates.append(info)
            best = max(candidates, key=lambda c: (c['inner_score'], -c['parameters']))
            logits, info = fit(x, meta, labels, train, None, best['config'], best['best_epoch'],
                               directory / 'outer', deadline, seed)
            metrics, predictions, probabilities = research.predict_metrics(logits, labels, val)
            delta = ((predictions[val] == labels[val]).float() - (base_pred[val] == labels[val]).float()) @ torch.tensor(data.WEIGHTS)
            result = {'metrics': metrics, 'candidates': candidates, **info,
                      'vs_foot': sparse.source_interval(delta.numpy(), rows.iloc[val].source_id)}
            records = [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                        'prediction': predictions[i].tolist(), 'probabilities': [p[i].tolist() for p in probabilities]} for i in val]
            data.dump(directory / 'predictions.json', records); data.dump(directory / 'result.json', result)
            results[variant] = result
            print(json.dumps({'stage': variant, 'score': metrics['weighted_score']}), flush=True)
        data.dump(out / 'summary.json', results)
        data.dump(out / 'status.json', {'stage': 'complete', 'seconds': time.monotonic() - start})
    except Exception as exc:
        data.dump(out / 'status.json', {'stage': 'error', 'error': str(exc)}); raise
    finally:
        data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'train'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--folds', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'pack': print(json.dumps(pack(args.cache, args.folds, args.out), indent=2))
    else: execute(args.cache, args.out)
