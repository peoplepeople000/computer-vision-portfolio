"""R5: bounded GPU tree screen on verified cached features, separate targets."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import structured_probe as s
from pose_selection import select_pose

DEPTHS = (2, 4)
SEEDS = (6643, 6644, 6645)


def choose_candidate(baseline_accuracy, candidates):
    chosen = {'kind': 'foot_rule', 'inner_accuracy': baseline_accuracy}
    for candidate in candidates:
        if candidate['inner_accuracy'] > chosen['inner_accuracy']:
            chosen = {'kind': 'tree', **candidate}
    return chosen


def inputs(base, image_bundle, geometry_bundle):
    source = base / 'r3_classification'
    manifest = json.loads((source / 'artifact_manifest.json').read_text())
    assert s.data.sha(source / 'features.npz') == manifest['features.npz']
    a = np.load(source / 'features.npz')
    rows = s.pd.read_csv(image_bundle / 'public_folds.csv')
    np.testing.assert_array_equal(a['clips'], rows['clip'])
    original = json.loads((image_bundle / 'manifest.json').read_text())
    assert s.data.sha(image_bundle / 'public_folds.csv') == original['files']['public_folds.csv']
    poses = a['pose_motion_view'][:, :552].reshape(len(rows), 3, 2, 23, 4)
    latest = []
    for i, record in enumerate(original['records']):
        assert record['clip'] == rows.iloc[i]['clip']
        latest.append(select_pose(poses[i], record['mask'], record['timestamps'], True)[0])
    gm = json.loads((geometry_bundle / 'manifest.json').read_text())
    assert s.data.sha(geometry_bundle / 'geometry.npz') == gm['files']['geometry.npz']
    geometry = np.load(geometry_bundle / 'geometry.npz')
    np.testing.assert_array_equal(geometry['clips'], rows['clip'])
    arrays = {'pose_last': a['body_foot_last'], 'pose_latest': np.stack(latest),
              'geometry': geometry['relative_last']}
    arrays['pose_geometry'] = np.c_[arrays['pose_latest'], arrays['geometry']]
    meta = torch.nn.functional.one_hot(torch.tensor([{'left': 0, 'right': 1}.get(f, 2) for f in rows.preferred_foot]), 3).float()
    labels = torch.tensor([[cs.index(row[k]) for k, cs in s.data.TARGETS.items()] for _, row in rows.iterrows()])
    arrays = {k: np.c_[v, meta.numpy()].astype(np.float32) for k, v in arrays.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    return rows, arrays, meta, labels


def gpu_config(booster):
    config = json.loads(booster.save_config())
    assert config['learner']['generic_param']['device'].startswith('cuda'), config
    assert 'grow_gpu_hist' in json.dumps(config), 'GPU histogram updater required'
    return config


def train(x, y, prior, fit_indices, check_indices, classes, depth, rounds, seed, out):
    import xgboost as xgb
    config = {'depth': depth, 'rounds': rounds, 'seed': seed, 'classes': classes,
              'fit_indices': fit_indices.tolist(), 'check_indices': None if check_indices is None else check_indices.tolist(),
              'source_sha256': s.data.sha(__file__), 'xgboost': xgb.__version__}
    s.data.immutable_json(out / 'protocol.json', config)
    matrix = xgb.DMatrix(x, label=y, base_margin=prior)
    fit = matrix.slice(fit_indices)
    if (out / 'result.json').exists():
        info = json.loads((out / 'result.json').read_text())
        assert s.data.sha(out / 'model.ubj') == info['model_sha256']
        booster = xgb.Booster(model_file=out / 'model.ubj'); booster.set_param({'device': 'cuda'})
        gpu_config(booster)
        return booster.predict(matrix, iteration_range=(0, info['selected_rounds'])), info
    params = {'objective': 'multi:softprob', 'num_class': classes, 'eval_metric': 'merror',
              'device': 'cuda', 'tree_method': 'hist', 'max_depth': depth, 'eta': .05,
              'min_child_weight': 5, 'lambda': 10, 'subsample': .85, 'colsample_bytree': .85,
              'seed': seed, 'nthread': 4}
    evaluation = [(fit, 'fit')]
    if check_indices is not None: evaluation.append((matrix.slice(check_indices), 'inner'))
    started = time.monotonic(); history = {}
    booster = xgb.train(params, fit, num_boost_round=rounds, evals=evaluation,
                        early_stopping_rounds=30 if check_indices is not None else None,
                        evals_result=history, verbose_eval=False)
    selected = booster.best_iteration + 1 if check_indices is not None else rounds
    runtime = gpu_config(booster)
    probabilities = booster.predict(matrix, iteration_range=(0, selected))
    booster.save_model(out / 'model.ubj')
    s.data.dump(out / 'runtime_config.json', runtime)
    s.data.dump(out / 'history.json', history)
    np.save(out / 'train_only_prior.npy', prior)
    reloaded = xgb.Booster(model_file=out / 'model.ubj'); reloaded.set_param({'device': 'cuda'})
    np.testing.assert_allclose(probabilities, reloaded.predict(matrix, iteration_range=(0, selected)), atol=1e-7, rtol=1e-6)
    info = {'selected_rounds': selected, 'trained_rounds': booster.num_boosted_rounds(),
            'inner_accuracy': None if check_indices is None else float((probabilities[check_indices].argmax(1) == y[check_indices]).mean()),
            'seconds': time.monotonic() - started, 'model_sha256': s.data.sha(out / 'model.ubj'),
            'depth': depth, 'cuda_histogram_verified': True}
    s.data.dump(out / 'result.json', info)
    print(json.dumps({'fit': str(out), **info}), flush=True)
    return probabilities, info


def profile(base, image_bundle, geometry_bundle, out):
    import xgboost as xgb
    assert torch.cuda.is_available()
    rows, arrays, meta, labels = inputs(base, image_bundle, geometry_bundle)
    fit = np.flatnonzero(~rows.fold.isin([0, 1]))
    prior = s.research.foot_priors(meta, labels, torch.tensor(fit)).numpy()[:, :3]
    _, info = train(arrays['pose_geometry'], labels[:, 0].numpy(), prior, fit, None, 3, 4, 20, 6643, out / 'cost_pilot')
    report = {'gpu': torch.cuda.get_device_name(0), 'xgboost': xgb.__version__, 'fit_folds': [2, 3, 4],
              'seconds_20_rounds_including_io': info['seconds'], 'estimated_max_12000_rounds_seconds': info['seconds'] * 600,
              'cuda_histogram_verified': info['cuda_histogram_verified'], 'note': 'Cost pilot model discarded; never used in selected fits.'}
    s.data.dump(out / 'profile.json', report); print(json.dumps(report), flush=True)
    return report


def execute(base, image_bundle, geometry_bundle, out, budget=900):
    import xgboost as xgb
    assert torch.cuda.is_available()
    rows, arrays, meta, labels = inputs(base, image_bundle, geometry_bundle)
    train_ix, val, inner, check = [np.flatnonzero(a) for a in (rows.fold != 0, rows.fold == 0, ~rows.fold.isin([0, 1]), rows.fold == 1)]
    assert set(rows.iloc[train_ix].source_id).isdisjoint(rows.iloc[val].source_id)
    assert set(rows.iloc[inner].source_id).isdisjoint(rows.iloc[check].source_id)
    protocol = {'source_sha256': s.data.sha(__file__), 'xgboost': xgb.__version__, 'variants': list(arrays),
                'depths': DEPTHS, 'seeds': SEEDS, 'inner_fold': 1, 'outer_fold': 0, 'max_rounds': 200,
                'early_stopping_rounds': 30, 'selection': 'Each target independently; inner accuracy, foot rule wins ties.',
                'feature_manifest_sha256': s.data.sha(base / 'r3_classification/artifact_manifest.json'),
                'geometry_manifest_sha256': s.data.sha(geometry_bundle / 'manifest.json'),
                'folds_sha256': s.data.sha(image_bundle / 'public_folds.csv'),
                'gpu': torch.cuda.get_device_name(0)}
    s.data.immutable_json(out / 'protocol.json', json.loads(json.dumps(protocol)))
    priors = {name: s.research.foot_priors(meta, labels, torch.tensor(ix)).numpy() for name, ix in [('inner', inner), ('outer', train_ix)]}
    starts = (0, 3, 6); counts = (3, 3, 2)
    baseline = np.column_stack([priors['outer'][:, start:start+n].argmax(1) for start, n in zip(starts, counts)])
    started = time.monotonic(); summary = {}
    def boundary():
        if time.monotonic() - started > budget - 60: raise TimeoutError('Completed tree fits saved; resume skips verified models.')
    try:
        for variant, x in arrays.items():
            choices = []
            for k, (start, n) in enumerate(zip(starts, counts)):
                margin = priors['inner'][:, start:start+n]
                baseline_accuracy = float((margin[check].argmax(1) == labels[check, k].numpy()).mean())
                candidates = []
                for depth in DEPTHS:
                    boundary()
                    _, info = train(x, labels[:, k].numpy(), margin, inner, check, n, depth, 200, 6643,
                                    out / variant / f'inner_target{k}_depth{depth}')
                    candidates.append(info)
                chosen = choose_candidate(baseline_accuracy, candidates)
                choices.append({'selected': chosen, 'candidates': candidates})
            trials, deltas = [], []
            for seed in SEEDS:
                probabilities = []
                for k, (start, n) in enumerate(zip(starts, counts)):
                    choice = choices[k]['selected']; margin = priors['outer'][:, start:start+n]
                    if choice['kind'] == 'foot_rule':
                        probabilities.append(torch.softmax(torch.from_numpy(margin), 1).numpy()); continue
                    boundary()
                    probs, _ = train(x, labels[:, k].numpy(), margin, train_ix, None, n, choice['depth'], choice['selected_rounds'], seed,
                                     out / variant / str(seed) / f'target{k}')
                    probabilities.append(probs)
                pred = np.column_stack([p.argmax(1) for p in probabilities])
                metrics = s.model.metric_report(labels[val].numpy(), pred[val])
                records = [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(), 'prediction': pred[i].tolist(),
                            'probabilities': [p[i].tolist() for p in probabilities]} for i in val]
                s.data.dump(out / variant / str(seed) / 'predictions.json', records)
                trials.append({'seed': seed, **metrics})
                deltas.append(((pred[val] == labels[val].numpy()).astype(float) - (baseline[val] == labels[val].numpy())) @ np.array(s.data.WEIGHTS))
            summary[variant] = {'choices': choices, 'trials': trials, 'mean': float(np.mean([t['weighted_score'] for t in trials])),
                                'sd': float(np.std([t['weighted_score'] for t in trials], ddof=1)),
                                'vs_foot': s.sparse.source_interval(np.mean(deltas, 0), rows.iloc[val].source_id)}
            s.data.dump(out / 'partial_summary.json', summary)
            print(json.dumps({'variant': variant, 'mean': summary[variant]['mean']}), flush=True)
        s.data.dump(out / 'summary.json', summary)
        s.data.dump(out / 'status.json', {'stage': 'complete', 'seconds': time.monotonic() - started})
    except Exception as error:
        s.data.dump(out / 'status.json', {'stage': 'error', 'error': str(error)}); raise
    finally:
        s.data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): s.data.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    return summary
