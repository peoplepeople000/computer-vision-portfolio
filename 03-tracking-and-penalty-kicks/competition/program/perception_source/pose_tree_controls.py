"""R16 controls from the teammate reference, without new perception models."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pose_mirror
import rgb_gru_data as data


def canonicalize(values, labels):
    """Map observed left-foot kicks to right-foot coordinates; keep unknowns."""
    values, labels = np.asarray(values), np.asarray(labels)
    assert values.ndim == 2 and values.shape[1] == 189 and labels.shape == (len(values), 3)
    metadata = values[:, 186:]
    assert np.isin(metadata, [0, 1]).all() and (metadata.sum(1) == 1).all()
    flip = metadata[:, 0] == 1
    transformed, targets = values.copy(), labels.copy()
    transformed[flip] = pose_mirror.mirror_features(values[flip])
    targets[flip] = pose_mirror.mirror_labels(labels[flip])
    return transformed, targets, flip


def direction_basis(values, flip):
    """An involution for direction probabilities/logits; outcomes stay unchanged."""
    result = np.asarray(values).copy(); flip = np.asarray(flip, bool)
    assert result.ndim == 2 and len(flip) == len(result) and result.shape[1] in (2, 3)
    if result.shape[1] == 3: result[flip] = result[flip][:, [2, 1, 0]]
    return result


def source_weights(sources, fit_indices, balanced):
    """Fit-source counts only. Equal total source mass, unchanged total clip mass."""
    fit_indices = np.asarray(fit_indices, int); sources = np.asarray(sources)[fit_indices]
    assert len(fit_indices) and len(np.unique(fit_indices)) == len(fit_indices)
    if not balanced: return np.ones(len(fit_indices), np.float32)
    _, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return (weights * len(weights) / weights.sum()).astype(np.float32)


def make_variant(values, labels, margin, sources, fit_indices, variant):
    """Return fitting rows without ever augmenting a held-out row into training."""
    assert variant in ('mirror_control', 'mirror_source', 'canonical', 'catboost_mirror')
    n = len(values); fit_indices = np.asarray(fit_indices, int)
    assert labels.shape == (n, 3) and margin.shape == (n, 8) and len(sources) == n
    assert np.all((fit_indices >= 0) & (fit_indices < n))
    weights = source_weights(sources, fit_indices, variant == 'mirror_source')
    if variant == 'canonical':
        x, y, flip = canonicalize(values, labels)
        offsets = np.concatenate([direction_basis(margin[:, start:start + count], flip)
                                  for start, count in [(0, 3), (3, 3), (6, 2)]], 1)
        return x, y, offsets, fit_indices, weights, flip
    x = np.concatenate((values, pose_mirror.mirror_features(values)))
    y = np.concatenate((labels, pose_mirror.mirror_labels(labels)))
    offsets = np.concatenate((margin, margin[:, [2, 1, 0, 5, 4, 3, 6, 7]]))
    return x, y, offsets, np.r_[fit_indices, fit_indices + n], np.r_[weights / 2, weights / 2], np.zeros(n, bool)


def softmax(values):
    values = values - values.max(1, keepdims=True)
    exp = np.exp(values); return exp / exp.sum(1, keepdims=True)


def load_inputs(archive, folds):
    """Validate immutable cached inputs; this function performs no model compute."""
    import io
    import zipfile
    import pandas as pd
    import torch
    import research_loop
    archive, folds = Path(archive), Path(folds)
    assert data.sha(archive) == 'dd675833d1c43a1d63559cfcb0e85c2e857d8ea4d42bfd0a115d791430e63831'
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('r10_matched_manifest.json'))
        expected = json.loads(z.read('results/r10_classification/artifact_manifest.json'))
        payload = z.read('results/r10_classification/features.npz')
    assert hashlib.sha256(payload).hexdigest() == expected['features.npz']
    assert data.sha(folds) == manifest['files']['public_folds.csv']
    rows = pd.read_csv(folds)
    assert len(rows) == 1299 and rows['clip'].is_unique and '0526_public.mp4' not in set(rows['clip'])
    assert rows.groupby('source_id').fold.nunique().max() == 1
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        np.testing.assert_array_equal(rows['clip'], z['clips'])
        arrays = {s: z[s + '__new_pose'].astype(np.float32) for s in ('inner', 'outer')}
    metadata = np.eye(3, dtype=np.float32)[[{'left': 0, 'right': 1}.get(f, 2) for f in rows.preferred_foot]]
    for x in arrays.values():
        assert x.shape == (len(rows), 189) and np.isfinite(x).all()
        np.testing.assert_array_equal(x[:, 186:], metadata)
    labels = np.array([[classes.index(row[k]) for k, classes in data.TARGETS.items()] for _, row in rows.iterrows()])
    indices = {k: np.flatnonzero(mask) for k, mask in {
        'inner_fit': ~rows.fold.isin([0, 1]), 'inner_check': rows.fold == 1,
        'outer_fit': rows.fold != 0, 'outer_check': rows.fold == 0}.items()}
    margins = {s: research_loop.foot_priors(torch.from_numpy(metadata), torch.from_numpy(labels),
                 torch.from_numpy(indices[s + '_fit'])).numpy() for s in arrays}
    return rows, arrays, labels, margins, indices


def promotion(inner, delta=.005):
    """Select once using inner scores only; tied candidates keep declared order."""
    names = ('mirror_source', 'canonical', 'catboost_mirror')
    control = inner['mirror_control']['mean']
    candidates = [name for name in names if inner[name]['mean'] >= control + delta]
    return max(candidates, key=lambda name: inner[name]['mean']) if candidates else None


def fit_catboost(x, y, margin, fit_indices, weights, out, seed=6643, rounds=450):
    """GPU training; bounded CPU replay for unsupported multiclass GPU evaluation."""
    import torch
    import catboost
    from catboost import CatBoostClassifier, Pool
    assert torch.cuda.is_available(), 'CUDA GPU required; do not fall back to CPU'
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    params = {'iterations': rounds, 'depth': 4, 'learning_rate': .035, 'l2_leaf_reg': 15,
              'loss_function': 'MultiClass', 'classes_count': margin.shape[1], 'random_seed': seed,
              'task_type': 'GPU', 'devices': '0', 'use_best_model': False, 'verbose': False,
              'thread_count': 4, 'allow_writing_files': True, 'train_dir': str(out / 'training'),
              'save_snapshot': True, 'snapshot_file': str((out / 'training/progress.cbsnapshot').resolve()),
              'snapshot_interval': 30}
    digest = lambda a: hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
    protocol = {'source_sha256': data.sha(__file__), 'params': params, 'catboost': catboost.__version__,
                'x_sha256': digest(x), 'y_sha256': digest(y), 'margin_sha256': digest(margin),
                'fit_indices': np.asarray(fit_indices).tolist(), 'weights': np.asarray(weights).tolist(),
                'selection': 'Fixed number of trees; no validation set, early stopping or best-iteration selection.'}
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        previous = json.loads(protocol_path.read_text())
        # The first Kaggle run saved this completed model before GPU replay failed.
        # Reuse only this audited source and identical inputs/parameters/library.
        assert previous['source_sha256'] == 'e866c02f2ce7c83dd48e67b68f2423588515e4111b2a9e71ebc3cf646932995c'
        assert (out / 'result.json').exists(), 'Only completed prior models may cross this replay-only repair'
        assert {k: v for k, v in previous.items() if k != 'source_sha256'} == {k: v for k, v in protocol.items() if k != 'source_sha256'}
    else:
        data.immutable_json(protocol_path, protocol)
    data.immutable_json(out / 'replay_protocol.json', {'source_sha256': data.sha(__file__),
                        'training_protocol_sha256': data.sha(protocol_path), 'prediction_device': 'CPU',
                        'reason': 'CatBoost multiclass GPU evaluation is unsupported; training remains GPU.'})
    net = CatBoostClassifier(**params); started = time.monotonic()
    result_path = out / 'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text()); assert data.sha(out / 'model.cbm') == saved['model_sha256']
        net.load_model(str(out / 'model.cbm'))
    else:
        pool = Pool(x[fit_indices], label=y[fit_indices], weight=weights, baseline=margin[fit_indices])
        net.fit(pool)
        assert net.tree_count_ == rounds and net.get_all_params()['task_type'] == 'GPU'
        temporary = out / 'model.tmp.cbm'; net.save_model(str(temporary)); temporary.replace(out / 'model.cbm')
        saved = {'model_sha256': data.sha(out / 'model.cbm'), 'trees': net.tree_count_,
                 'seconds': time.monotonic() - started, 'gpu_training': True}
        data.dump(out / 'history.json', net.get_evals_result()); data.dump(out / 'result.json', saved)
    # Plain arrays carry no Pool baseline: add the train-only prior explicitly once.
    raw = net.predict(x, prediction_type='RawFormulaVal', task_type='CPU', thread_count=4)
    assert raw.shape == margin.shape and np.isfinite(raw).all()
    replay = CatBoostClassifier(); replay.load_model(str(out / 'model.cbm'))
    np.testing.assert_allclose(raw, replay.predict(x, prediction_type='RawFormulaVal', task_type='CPU', thread_count=4), rtol=1e-6, atol=1e-7)
    probability = softmax(raw + margin)
    data.dump(out / 'reload_verification.json', {**saved, 'gpu_inference': False, 'inference_device': 'CPU',
              'probability_sha256': digest(probability), 'prior_sha256': digest(margin)})
    return probability, saved


VARIANTS = ('mirror_control', 'mirror_source', 'canonical', 'catboost_mirror')
SEEDS = (6643, 6644, 6645)
ROUNDS = (45, 19)


def save_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    data.dump(temporary, value); temporary.replace(path)


def execute(archive, folds, out, budget=1200, platform='Colab'):
    """Fixed inner controls, then at most one promoted branch and matched control."""
    import torch
    assert torch.cuda.is_available(), 'GPU required; no CPU model fallback'
    assert platform in ('Colab', 'Kaggle')
    import xgboost
    import catboost
    import rgb_gru
    import research_loop
    import sparse_role_comparison
    assert xgboost.__version__ == '3.4.1', 'Use the frozen R13 XGBoost version'
    torch.set_num_threads(4)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, arrays, labels, margins, ix = load_inputs(archive, folds)
    n = len(rows); started = time.monotonic()
    source_files = [__file__, pose_mirror.__file__, data.__file__, rgb_gru.__file__,
                    research_loop.__file__, sparse_role_comparison.__file__]
    protocol = {'sources': {Path(p).name: data.sha(p) for p in source_files},
                'archive_sha256': data.sha(archive), 'folds_sha256': data.sha(folds),
                'variants': list(VARIANTS), 'seeds': list(SEEDS), 'xgb_rounds': list(ROUNDS),
                'catboost_rounds': 450, 'inner_fold': 1, 'outer_fold': 0,
                'inner_promotion_delta': .005, 'xgboost': xgboost.__version__,
                'catboost': catboost.__version__, 'outcome': 'Unchanged train-only foot prior',
                'selection': 'Three-seed inner mean; one best branch at least .005 above matched control.',
                'note': 'Exploratory repeated folds; no competition public-score claim.'}
    data.immutable_json(out / 'protocol.json', protocol)
    save_json(out / 'hardware.json', {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                                     'torch': str(torch.__version__), 'platform': platform})
    report = {'stage': 'running', 'inner': {}, 'outer': {}}

    def boundary(reserve=60):
        if time.monotonic() - started > budget - reserve:
            raise TimeoutError('Budget boundary; completed fits and CatBoost snapshots saved')

    def variant_inputs(stage, variant):
        return make_variant(arrays[stage], labels, margins[stage], rows.source_id,
                            ix[stage + '_fit'], variant)

    def fit_trial(stage, variant, seed):
        x, y, m, fit, weights, flip = variant_inputs(stage, variant)
        probabilities = []; infos = []
        target_dir = out / stage / variant / str(seed)
        for k, (start, count) in enumerate(((0, 3), (3, 3), (6, 2))):
            boundary(); offset = m[:, start:start + count]
            if k == 2:
                p = softmax(offset); info = {'kind': 'foot_rule'}
            elif variant == 'catboost_mirror':
                p, info = fit_catboost(x, y[:, k], offset, fit, weights, target_dir / f'target{k}', seed)
            else:
                p, info = pose_mirror.fit(x, y[:, k], offset, fit, None, weights, count,
                                         4, ROUNDS[k], seed, target_dir / f'target{k}')
            p = direction_basis(p[:n], flip)
            assert np.isfinite(p).all(); np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
            probabilities.append(p); infos.append(info)
        check = ix[stage + '_check']; predicted = np.column_stack([p.argmax(1) for p in probabilities])
        metrics = rgb_gru.metric_report(labels[check], predicted[check])
        records = [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                    'prediction': predicted[i].tolist(), 'probabilities': [p[i].tolist() for p in probabilities]}
                   for i in check]
        save_json(target_dir / 'predictions.json', records)
        trial = {'seed': seed, **metrics, 'fits': infos}
        save_json(target_dir / 'metrics.json', trial)
        print(json.dumps({'stage': stage, 'variant': variant, 'seed': seed,
                          'weighted_score': metrics['weighted_score']}), flush=True)
        return trial, predicted[check], [p[check] for p in probabilities]

    def finish_trials(stage, variant):
        trials, predicted, probabilities = [], [], []
        for seed in SEEDS:
            trial, pred, prob = fit_trial(stage, variant, seed)
            if stage == 'inner' and variant == 'mirror_control' and seed == 6643:
                # Establish the comparison before fitting any new branch.
                expected = np.array([155, 232, 196]) / 262
                actual = np.array([trial[key] for key in data.TARGETS])
                verified = bool(np.allclose(actual, expected, atol=1e-10, rtol=0))
                save_json(out / 'control_reproduction.json', {'passed': verified,
                          'expected_accuracy': expected.tolist(), 'actual_accuracy': actual.tolist()})
                assert verified, 'R13 control changed; inspect runtime/input reproducibility before comparing'
            trials.append(trial); predicted.append(pred); probabilities.append(prob)
        scores = [t['weighted_score'] for t in trials]
        ensemble = np.column_stack([np.mean([p[k] for p in probabilities], 0).argmax(1) for k in range(3)])
        metrics = {'trials': trials, 'mean': float(np.mean(scores)), 'sd': float(np.std(scores, ddof=1)),
                   'probability_ensemble': rgb_gru.metric_report(labels[ix[stage + '_check']], ensemble)}
        return metrics, np.stack(predicted)

    try:
        # Reproduce the familiar XGBoost control before the new learner cost gate.
        report['inner']['mirror_control'], _ = finish_trials('inner', 'mirror_control')
        save_json(out / 'summary.json', report)
        x, y, m, fit, weights, _ = variant_inputs('inner', 'catboost_mirror')
        boundary()
        _, warm = fit_catboost(x, y[:, 0], m[:, :3], fit, weights, out / 'discarded_warmup', rounds=10)
        boundary()
        _, pilot = fit_catboost(x, y[:, 0], m[:, :3], fit, weights, out / 'discarded_cost_pilot', rounds=30)
        # Six inner fits plus at most six outer fits, with larger outer rows and saving reserve.
        estimate = float(warm['seconds'] + pilot['seconds'] * (450 / 30) * 12 * 2)
        cost = {'warmup_seconds': warm['seconds'], 'seconds_30_trees': pilot['seconds'],
                'estimated_catboost_total_seconds': estimate, 'gate_passed': estimate < 900,
                'gpu_training_and_cpu_reload_verified': True}
        save_json(out / 'cost_pilot.json', cost)
        if not cost['gate_passed']:
            report.update(stage='cost_rejected', cost=cost)
            return report
        for variant in VARIANTS[1:]:
            report['inner'][variant], _ = finish_trials('inner', variant)
            save_json(out / 'summary.json', report)
        winner = promotion(report['inner'])
        report.update(stage='inner_complete', promoted=winner)
        save_json(out / 'inner_summary.json', report)
        if winner:
            outer_predictions = {}
            for variant in ('mirror_control', winner):
                report['outer'][variant], outer_predictions[variant] = finish_trials('outer', variant)
                save_json(out / 'summary.json', report)
            truth = labels[ix['outer_check']]
            delta = np.mean((outer_predictions[winner] == truth).astype(float) -
                            (outer_predictions['mirror_control'] == truth).astype(float), axis=0) @ np.array(data.WEIGHTS)
            report['paired_vs_control'] = sparse_role_comparison.source_interval(delta, rows.iloc[ix['outer_check']].source_id)
        report.update(stage='complete', seconds=time.monotonic() - started)
        return report
    except Exception as error:
        report.update(stage='interrupted' if isinstance(error, TimeoutError) else 'failed',
                      error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        raise
    finally:
        save_json(out / 'summary.json', report)
        save_json(out / 'artifact_manifest.json', {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*')
                  if p.is_file() and p.name not in ('artifact_manifest.json', 'console.log', 'process.json')
                  and p.suffix != '.tmp'})
