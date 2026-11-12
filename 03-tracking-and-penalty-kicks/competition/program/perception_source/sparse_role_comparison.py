"""Paired scene/static-role/best-snapshot GRU trials on one frozen sparse cache."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import backbone_diagnostics as diagnostics
import rgb_gru as model
import rgb_gru_data as data

VARIANTS = ('scene_tail3', 'static_roles3', 'best_roles1')
SEEDS = (6643, 6644, 6645)


def pack_bundle(cache, folds, out):
    """Consolidate verified features without changing any model input values."""
    cache, folds, out = Path(cache), Path(folds), Path(out)
    rows, masks, hashes = audit(cache, folds)
    items = list(model.ClipDataset(rows, cache, 'features', 'all'))
    batch = model.collate(items)
    best = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(masks[i['clip']]) for i in items], batch_first=True)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'features.npz', clips=np.asarray(batch['clips']),
                        best_view_mask=best.numpy(), **{k: batch[k].numpy() for k in
                        ['inputs', 'view_mask', 'timestamps', 'lengths', 'metadata', 'labels']})
    shutil.copy2(folds, out / 'public_folds.csv')
    manifest = {'version': 1, 'clips': len(rows), 'cache_manifest_sha256': data.sha(cache / 'artifact_manifest.json'),
                'cache_protocol_sha256': data.sha(cache / 'protocol.json'),
                'source_protocol': json.loads((cache / 'protocol.json').read_text()),
                'feature_hashes': hashes,
                'files': {name: data.sha(out / name) for name in ['features.npz', 'public_folds.csv']}}
    data.dump(out / 'bundle_manifest.json', manifest)
    _, restored_masks, _, restored, _ = load_bundle(out, out / 'public_folds.csv')
    for item in items:
        for key in ['inputs', 'view_mask', 'timestamps', 'metadata', 'labels']:
            torch.testing.assert_close(restored[item['clip']][key], item[key], rtol=0, atol=0)
        np.testing.assert_array_equal(restored_masks[item['clip']], masks[item['clip']])
    report = {'clips_verified': len(items), 'exact_model_input_parity': True,
              'bundle_sha256': data.sha(out / 'features.npz'),
              'bundle_bytes': (out / 'features.npz').stat().st_size}
    data.dump(out / 'bundle_verification.json', report)
    return report


def load_bundle(root, folds):
    root, folds = Path(root), Path(folds)
    manifest = json.loads((root / 'bundle_manifest.json').read_text())
    assert manifest['version'] == 1 and manifest['clips'] == 1299
    assert set(manifest['files']) == {'features.npz', 'public_folds.csv'}
    for name, digest in manifest['files'].items():
        assert data.sha(root / name) == digest, name
    assert data.sha(folds) == manifest['files']['public_folds.csv']
    protocol = manifest['source_protocol']
    assert protocol['encoder']['weights'] == 'IMAGENET1K_V1'
    assert protocol['source_sha256']['rgb_gru.py'] == data.sha(model.__file__)
    assert protocol['source_sha256']['rgb_gru_data.py'] == data.sha(data.__file__)
    assert protocol['models']['data/processed/rgb_gru_v1/public_folds.csv']['sha256'] == data.sha(folds)
    rows = pd.read_csv(folds)
    assert len(rows) == rows['clip'].nunique() == 1299 and rows.groupby('source_id').fold.nunique().max() == 1
    assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    with np.load(root / 'features.npz', allow_pickle=False) as packed:
        arrays = {k: packed[k] for k in packed.files}
    np.testing.assert_array_equal(arrays['clips'], rows['clip'].to_numpy())
    assert arrays['inputs'].shape == (1299, 3, 3, 1280)
    assert arrays['view_mask'].shape == arrays['best_view_mask'].shape == (1299, 3, 3)
    assert arrays['timestamps'].shape == arrays['metadata'].shape == arrays['labels'].shape == (1299, 3)
    assert arrays['lengths'].shape == (1299,) and ((arrays['lengths'] >= 1) & (arrays['lengths'] <= 3)).all()
    assert np.isfinite(arrays['inputs']).all() and np.isfinite(arrays['timestamps']).all()
    expected = np.array([[values.index(row[k]) for k, values in data.TARGETS.items()] for _, row in rows.iterrows()])
    np.testing.assert_array_equal(arrays['labels'], expected)
    metadata = np.zeros((len(rows), 3), np.float32)
    for i, foot in enumerate(rows.preferred_foot):
        metadata[i, {'left': 0, 'right': 1}.get(foot, 2)] = 1
    np.testing.assert_array_equal(arrays['metadata'], metadata)
    items, masks = {}, {}
    for i, clip in enumerate(arrays['clips']):
        n = int(arrays['lengths'][i]); mask = arrays['view_mask'][i, :n]; best = arrays['best_view_mask'][i, :n]
        assert mask[:, 0].all() and best[:, 0].all() and not (best & ~mask).any()
        assert best[:, 1:].any(1).sum() <= 1 and (np.diff(arrays['timestamps'][i, :n]) > 0).all()
        items[str(clip)] = {'clip': str(clip), **{k: torch.from_numpy(arrays[k][i, :n]) for k in
                            ['inputs', 'view_mask', 'timestamps']}, **{k: torch.from_numpy(arrays[k][i]) for k in
                            ['metadata', 'labels']}}
        masks[str(clip)] = best
    provenance = {k: manifest[k] for k in ['cache_manifest_sha256', 'cache_protocol_sha256']}
    provenance['bundle_manifest_sha256'] = data.sha(root / 'bundle_manifest.json')
    provenance['bundle_sha256'] = manifest['files']['features.npz']
    return rows, masks, manifest['feature_hashes'], items, provenance


def source_interval(delta, sources, draws=10000):
    """Paired source-cluster bootstrap, retaining clip-weighted scoring."""
    delta, sources = np.asarray(delta, float), np.asarray(sources)
    if delta.ndim != 1 or len(delta) != len(sources) or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Expected finite paired differences and matching sources')
    groups = [np.flatnonzero(sources == s) for s in sorted(set(sources))]
    sums = np.array([delta[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    sample = np.random.default_rng(6643).integers(0, len(groups), (draws, len(groups)))
    values = sums[sample].sum(1) / counts[sample].sum(1)
    return {'delta': float(delta.mean()), 'source_cluster_95pct': np.quantile(values, [.025, .975]).tolist(),
            'source_count': len(groups)}


def verify_results(out, folds):
    """Verify saved trials and compute an exploratory report without fitting models."""
    out, folds = Path(out), Path(folds)
    manifest = json.loads((out / 'artifact_manifest.json').read_text())
    for relative, digest in manifest.items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        assert data.sha(out / path) == digest, relative
    protocol = json.loads((out / 'comparison_protocol.json').read_text())
    assert data.sha(folds) == protocol['fold_sha256']
    assert protocol['variants'] == list(VARIANTS) and protocol['seeds'] == list(SEEDS)
    status = json.loads((out / 'status.json').read_text())
    expected_runs = {f'{v}_{s}' for v in VARIANTS for s in SEEDS}
    assert status['stage'] == 'complete' and set(status['experiments']) == expected_runs
    rows = pd.read_csv(folds); train, val = rows[rows.fold != 0], rows[rows.fold == 0]
    assert len(rows) == 1299 and set(train.source_id).isdisjoint(val.source_id)
    truth = np.array([[labels.index(row[k]) for k, labels in data.TARGETS.items()] for _, row in val.iterrows()])
    global_labels = [labels.index(train[k].mode().iloc[0]) for k, labels in data.TARGETS.items()]
    by_foot = {foot: [labels.index(group[k].mode().iloc[0]) for k, labels in data.TARGETS.items()]
               for foot, group in train.groupby('preferred_foot')}
    foot_labels = np.array([by_foot.get(foot, global_labels) for foot in val.preferred_foot])
    foot_scores = (truth == foot_labels) @ np.array(data.WEIGHTS)
    scores, experiments = {}, {}
    for seed in SEEDS:
        initial = set()
        for variant in VARIANTS:
            name = f'{variant}_{seed}'; run = out / name
            summary = json.loads((run / 'summary.json').read_text())
            best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
            config = best['config']
            assert config == last['config'] and config['variant'] == variant and config['seed'] == seed
            assert config['fold_sha256'] == protocol['fold_sha256']
            assert config['cache_manifest_sha256'] == protocol['cache_manifest_sha256']
            assert set(config['train_clips']) == set(train['clip']) and set(config['validation_clips']) == set(val['clip'])
            assert last['epoch'] == len(last['history']) == 8 and last['optimizer']['state']
            assert {'python', 'numpy', 'torch', 'loader', 'cuda'} <= set(last['rng'])
            assert summary['checkpoint_sha256'] == data.sha(run / 'best.pt')
            history = json.loads((run / 'history.json').read_text())
            assert history == last['history']
            assert best['epoch'] == summary['best_epoch'] == 1 + int(np.argmax([h['validation']['weighted_score'] for h in history]))
            pred = pd.DataFrame(json.loads((run / 'validation_predictions.json').read_text()))
            assert not pred['clip'].duplicated().any() and set(pred['clip']) == set(val['clip'])
            pred = pred.set_index('clip').loc[val['clip']]
            encoded = np.array([[labels.index(row[k]) for k, labels in data.TARGETS.items()] for _, row in pred.iterrows()])
            for i, key in enumerate(data.TARGETS):
                probabilities = np.array(pred[key + '_probabilities'].tolist())
                assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
                np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
                np.testing.assert_array_equal(probabilities.argmax(1), encoded[:, i])
            metrics = model.metric_report(truth, encoded)
            for key in [*data.TARGETS, 'weighted_score']:
                assert abs(metrics[key] - summary['metrics'][key]) < 1e-12
                assert abs(metrics[key] - best['metrics'][key]) < 1e-12
            assert metrics['confusion_matrices'] == summary['metrics']['confusion_matrices']
            scores[name] = (truth == encoded) @ np.array(data.WEIGHTS)
            initial.add(summary['initial_state_sha256'])
            assert summary['initial_state_sha256'] == best['initial_state_sha256']
            experiments[name] = {'metrics': metrics, 'best_epoch': best['epoch'],
                                 'train_score_at_best': history[best['epoch'] - 1]['train']['weighted_score']}
            del best, last
        assert len(initial) == 1
    means = {v: np.mean([scores[f'{v}_{s}'] for s in SEEDS], axis=0) for v in VARIANTS}
    aggregates = {}
    for variant in VARIANTS:
        values = [scores[f'{variant}_{s}'].mean() for s in SEEDS]
        aggregates[variant] = {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)),
            'vs_scene': source_interval(means[variant] - means['scene_tail3'], val.source_id),
            'vs_foot': source_interval(means[variant] - foot_scores, val.source_id),
            'paired_seed_differences': [float((scores[f'{variant}_{s}'] - scores[f'scene_tail3_{s}']).mean()) for s in SEEDS]}
    return {'verified_files': len(manifest), 'experiments': experiments, 'aggregates': aggregates,
            'train_clips': len(train), 'validation_clips': len(val), 'foot_reference': float(foot_scores.mean()),
            'per_clip': [{'clip': c, 'source_id': source, **{v: float(means[v][i]) for v in VARIANTS}}
                         for i, (c, source) in enumerate(zip(val['clip'], val.source_id))],
            'limitations': 'Exploratory fold 0; validation-selected epochs. Cluster intervals are conditional on these three fitted seeds, not independent folds or multiple-comparison-adjusted confidence claims.'}


def with_views(item, variant, best_mask):
    """Change visibility only; retain identical images, timestamps and metadata."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    result = dict(item)
    mask = item['view_mask'].clone()
    if variant == 'scene_tail3':
        mask[:, 1:] = False
    elif variant == 'best_roles1':
        best = torch.as_tensor(best_mask, dtype=torch.bool)
        assert best.shape == mask.shape and not (best & ~mask).any()
        assert best[:, 0].all() and best[:, 1:].any(1).sum() <= 1
        mask = best.clone()
    result['view_mask'] = mask
    return result


def audit(cache, folds):
    rows = pd.read_csv(folds)
    assert len(rows) == rows['clip'].nunique() == 1299
    assert rows.groupby('source_id').fold.nunique().max() == 1
    assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    protocol = json.loads((cache / 'protocol.json').read_text())
    assert data.sha(folds) == protocol['models']['data/processed/rgb_gru_v1/public_folds.csv']['sha256']
    assert protocol['encoder']['weights'] == 'IMAGENET1K_V1'
    assert protocol['source_sha256']['rgb_gru.py'] == data.sha(model.__file__)
    assert protocol['source_sha256']['rgb_gru_data.py'] == data.sha(data.__file__)
    status = json.loads((cache / 'status.json').read_text())
    assert status['stage'] == 'complete' and status['completed'] == 1299
    scope = pd.DataFrame(json.loads((cache / 'run_scope.json').read_text())['sample'])
    cols = ['clip', 'source_id', 'fold']
    pd.testing.assert_frame_equal(rows[cols].sort_values('clip').reset_index(drop=True),
                                  scope[cols].sort_values('clip').reset_index(drop=True))
    manifest = json.loads((cache / 'artifact_manifest.json').read_text())
    protocol_sha = data.sha(cache / 'protocol.json')
    assert manifest['protocol.json'] == protocol_sha
    masks, feature_hashes = {}, {}
    for clip in rows['clip']:
        record_name = f'records/{clip}.json'; feature_name = f'features/{clip}.npz'
        assert data.sha(cache / record_name) == manifest[record_name]
        record = json.loads((cache / record_name).read_text())
        assert record['inputs']['protocol_sha256'] == protocol_sha and record['cuda_inference_verified']
        digest = data.sha(cache / feature_name)
        assert digest == record['files'][feature_name] == manifest[feature_name]
        feature_hashes[clip] = digest
        with np.load(cache / feature_name, allow_pickle=False) as a:
            n = len(a['timestamps'])
            assert 1 <= n <= 3 and a['features'].shape == (n, 3, 1280)
            assert np.isfinite(a['features']).all() and np.isfinite(a['timestamps']).all()
            assert (np.diff(a['timestamps']) > 0).all() and (np.diff(a['frame_indices']) > 0).all()
            assert a['view_mask'][:, 0].all()
            masks[clip] = a['best_view_mask'].copy()
    return rows, masks, feature_hashes


def execute(args):
    assert torch.cuda.is_available(), 'CUDA required; no CPU training fallback'
    torch.set_num_threads(4)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_seconds - 120
    try:
        prep = time.monotonic()
        if (args.cache / 'bundle_manifest.json').exists():
            rows, best_masks, feature_hashes, items, provenance = load_bundle(args.cache, args.folds)
        else:
            rows, best_masks, feature_hashes = audit(args.cache, args.folds)
            items = {item['clip']: item for item in model.ClipDataset(rows, args.cache, 'features', 'all')}
            provenance = {'cache_protocol_sha256': data.sha(args.cache / 'protocol.json'),
                          'cache_manifest_sha256': data.sha(args.cache / 'artifact_manifest.json')}
        provenance = {**provenance,
                      'fold_sha256': data.sha(args.folds),
                      'code_sha256': {Path(p).name: data.sha(p) for p in
                                     [__file__, diagnostics.__file__, model.__file__, data.__file__]}}
        data.immutable_json(out / 'comparison_protocol.json', {
            **provenance, 'variants': list(VARIANTS), 'seeds': [6643, 6644, 6645],
            'epochs': 8, 'fold': 0, 'encoder_frozen': True, 'feature_hashes': feature_hashes,
            'note': 'Same frames, features, metadata, head architecture and paired initialization; only role masks differ.'})
        data.dump(out / 'runtime.json', {'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__)})
        print(json.dumps({'stage': 'inputs_ready', 'clips': len(items), 'seconds': time.monotonic() - prep,
                          'packed_input': 'bundle_sha256' in provenance}), flush=True)
        results = {}
        for seed in [6643, 6644, 6645]:
            initial = set()
            for variant in VARIANTS:
                if time.monotonic() > deadline:
                    raise diagnostics.BudgetStop('Before ' + variant)
                selected = {clip: with_views(item, variant, best_masks[clip]) for clip, item in items.items()}
                name = f'{variant}_{seed}'
                result = diagnostics.train_trial(
                    name, seed, False,
                    [selected[c] for c in rows.loc[rows.fold != 0, 'clip']],
                    [selected[c] for c in rows.loc[rows.fold == 0, 'clip']],
                    rows, out, deadline, {**provenance, 'variant': variant})
                results[name] = result
                initial.add(result['initial_state_sha256'])
            assert len(initial) == 1, 'Paired trials started from different head weights'
        data.dump(out / 'comparison.json', results)
        data.dump(out / 'status.json', {'stage': 'complete', 'experiments': list(results),
                                       'private_submission': False})
    except diagnostics.BudgetStop as exc:
        data.dump(out / 'status.json', {'stage': 'checkpointed', 'reason': str(exc)})
    except Exception as exc:
        data.dump(out / 'status.json', {'stage': 'error', 'reason': str(exc)})
        raise
    finally:
        data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*')
                  if p.is_file() and p.name != 'artifact_manifest.json' and p.suffix != '.tmp'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--verify-report', type=Path, help='Verify completed output and write report outside the immutable output root')
    args = parser.parse_args()
    if args.verify_report:
        if args.verify_report.resolve().is_relative_to(args.out.resolve()):
            parser.error('Verification report must be outside the immutable output root')
        report = verify_results(args.out, args.folds)
        data.dump(args.verify_report, report)
        print(json.dumps(report['aggregates'], indent=2))
    else:
        if args.cache is None:
            parser.error('--cache required for training')
        execute(args)
