"""R22 phase 1: source-held-out role selection from existing detections."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import role_selector as roles
import rgb_gru_data as data


def training_indices(frames, heldout):
    sources = {f['source_id'] for f in frames if f['fold'] == heldout}
    selected = [f for f in frames if f['fold'] != heldout and f['eligible_supervision']]
    assert sources.isdisjoint(f['source_id'] for f in selected)
    return np.array([i for f in selected for i in range(f['start'], f['end'])], int)


def crop_plan(frames, pairs, image_hashes):
    assert len(frames) == len(pairs)
    records, missing, reused = [], {}, set()
    for frame, pair in zip(frames, pairs):
        assert pair == [-1, -1] or (min(pair) >= 0 and max(pair) < len(frame['boxes']) and pair[0] != pair[1])
        keys = [None if j < 0 else f'{frame["clip"]}/{frame["anchor"]}_{j}.jpg' for j in pair]
        for role, (key, j) in enumerate(zip(keys, pair)):
            if key is None: continue
            if key in image_hashes: reused.add(key)
            else: missing[key] = {'clip': frame['clip'], 'anchor': frame['anchor'], 'timestamp': frame['timestamp'],
                                 'candidate': j, 'box': frame['boxes'][j]}
        records.append({'clip': frame['clip'], 'anchor': frame['anchor'], 'timestamp': frame['timestamp'],
                        'pair': pair, 'keys': keys, 'mask': [j >= 0 for j in pair],
                        'hashes': [image_hashes.get(k) if k else None for k in keys]})
    return records, missing, reused


def execute(bundle, out, heldout=2):
    import torch
    import xgboost as xgb
    import xgb_probe
    assert torch.cuda.is_available() and xgb.__version__ == '3.4.1'
    bundle, out = Path(bundle), Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items(): assert data.sha(bundle / name) == digest
    with np.load(bundle / 'candidates.npz') as z: x, y = z['features'], z['labels']
    frames = json.loads((bundle / 'frames.json').read_text())
    existing = json.loads((bundle / 'image-manifest.json').read_text())
    rows = pd.read_csv(bundle / 'public_folds.csv')
    assert len(rows) == 1299 and rows.groupby('source_id').fold.nunique().max() == 1
    assert '0526_public.mp4' not in set(rows['clip'])
    assert len(frames) == len(rows) * 3 and x.shape[1] == 30
    for i, f in enumerate(frames):
        assert f['clip'] == rows.iloc[i // 3]['clip'] and f['anchor'] == i % 3
        assert f['fold'] == rows.iloc[i // 3]['fold'] and f['source_id'] == rows.iloc[i // 3]['source_id']
    fit = training_indices(frames, heldout)
    assert np.isin(y[fit], [0, 1, 2]).all()
    counts = np.bincount(y[fit], minlength=3); assert (counts > 0).all()
    weights = np.sqrt(len(fit) / (3 * counts))
    params = {'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist',
        'max_depth': 5, 'eta': .05, 'min_child_weight': 3, 'lambda': 10, 'subsample': 1.,
        'colsample_bytree': 1., 'seed': 6643, 'nthread': 4, 'eval_metric': 'mlogloss'}
    protocol = {'source_sha256': data.sha(__file__), 'role_helper_sha256': data.sha(roles.__file__),
        'bundle_sha256': data.sha(bundle / 'manifest.json'), 'heldout_fold': heldout, 'params': params,
        'rounds': 1, 'fit_indices': fit.tolist(), 'class_weights': weights.tolist(), 'xgboost': xgb.__version__,
        'frozen_recipe': 'Depth5/one round from R9 selection; no retuning on held-out fold.',
        'scope': 'Point annotations supervise only train-source role ranker. No outcome labels in role features. No detector inference.'}
    data.immutable_json(out / 'protocol.json', protocol)
    matrix = xgb.DMatrix(x, label=np.maximum(y, 0)); model_path = out / 'model.ubj'; started = time.monotonic()
    if model_path.exists():
        previous = json.loads((out / 'fit.json').read_text()); assert data.sha(model_path) == previous['model_sha256']
        model = xgb.Booster(model_file=model_path); model.set_param({'device': 'cuda'})
    else:
        train = matrix.slice(fit); train.set_weight(weights[y[fit]])
        history = {}
        model = xgb.train(params, train, num_boost_round=1, evals=[(train, 'fit')], evals_result=history, verbose_eval=False)
        temporary = out / 'model.tmp.ubj'; model.save_model(temporary); temporary.replace(model_path)
        data.dump(out / 'history.json', history)
        data.dump(out / 'fit.json', {'model_sha256': data.sha(model_path), 'seconds': time.monotonic()-started})
    runtime = xgb_probe.gpu_config(model); probability = model.predict(matrix)
    replay = xgb.Booster(model_file=model_path); replay.set_param({'device': 'cuda'})
    xgb_probe.gpu_config(replay); np.testing.assert_allclose(probability, replay.predict(matrix), atol=1e-7, rtol=1e-6)
    pairs = [roles.choose_pair(probability[f['start']:f['end']]) for f in frames]
    records, missing, reused = crop_plan(frames, pairs, existing['image_hashes'])
    data.dump(out / 'selections.json', records); data.dump(out / 'missing_crops.json', missing)
    checks = [f for f in frames if f['fold'] == heldout and f['annotated']]
    coverage, _, _ = roles.coverage(probability, checks)
    np.savez_compressed(out / 'probabilities.npz', probabilities=probability)
    data.dump(out / 'runtime.json', {'gpu': torch.cuda.get_device_name(0), 'runtime_config': runtime, 'reload_verified': True})
    report = {'status': 'complete', 'phase': 'role_selection_and_reuse_audit', 'heldout_fold': heldout,
        'fit_sources': len({f['source_id'] for f in frames if f['fold'] != heldout and f['eligible_supervision']}),
        'heldout_sources': rows[rows.fold == heldout].source_id.nunique(), 'heldout_clips': int((rows.fold == heldout).sum()),
        'point_coverage': coverage, 'reusable_image_keys': len(reused), 'missing_image_keys': len(missing),
        'all_role_assignments': len(records) * 2, 'new_detector_calls': 0, 'new_pose_calls': 0,
        'model_sha256': data.sha(model_path), 'seconds': time.monotonic()-started,
        'next': 'Export only missing candidate crops, reuse RTMW/NLF by image hash, fit the fixed R19 comparison on new source split.',
        'limitation': 'Point coverage is not identity accuracy. Fold2 previously participated in research training; not a pristine test set.'}
    data.dump(out / 'summary.json', report); print(json.dumps(report), flush=True)
    return report
