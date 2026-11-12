"""R17: explicit 2D body relations from the frozen R10/R13 pose cache.

These are image-plane descriptors, not calibrated 3D rotations or measured
ball trajectories. Derive them AFTER mirror augmentation so anatomical roles,
actual kicking-foot metadata and signed directions remain consistent.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

VECTORS = (
    ('shoulder_line', 'support_shoulder', 'kick_shoulder'),
    ('hip_line', 'support_hip', 'kick_hip'),
    ('torso', 'hip_mid', 'shoulder_mid'),
    ('kick_thigh', 'kick_hip', 'kick_knee'),
    ('kick_shin', 'kick_knee', 'kick_ankle'),
    ('kick_foot', 'kick_heel', 'kick_toes'),
    ('support_thigh', 'support_hip', 'support_knee'),
    ('support_shin', 'support_knee', 'support_ankle'),
    ('support_foot', 'support_heel', 'support_toes'),
    ('stance', 'support_ankle', 'kick_ankle'),
    ('support_to_kick_toes', 'support_heel', 'kick_toes'),
)
ANGLES = (
    ('shoulder_line', 'hip_line'), ('hip_line', 'torso'),
    ('support_foot', 'kick_foot'), ('support_shin', 'support_foot'),
    ('kick_shin', 'kick_foot'), ('kick_thigh', 'kick_shin'),
    ('support_thigh', 'support_shin'), ('torso', 'support_shin'),
    ('torso', 'kick_shin'), ('stance', 'support_foot'),
)


def relational_features(values):
    """Return 117 kicker descriptors plus names; never inspect labels or folds."""
    values = np.asarray(values, np.float32)
    assert values.ndim == 2 and values.shape[1] == 189
    metadata = values[:, 186:189]
    assert np.isin(metadata, [0, 1]).all() and (metadata.sum(1) == 1).all()
    joints = values[:, :92].reshape(-1, 23, 4)
    known = metadata[:, 2] == 0
    left_kick = metadata[:, 0] == 1
    row = np.arange(len(values)); points = {}

    def point(indices, available):
        p = joints[row, indices]
        valid = available & (p[:, 3] > 0) & np.isfinite(p).all(1)
        return (np.where(valid[:, None], p[:, :2], 0),
                np.where(valid, np.clip(p[:, 2], 0, 1), 0), valid)

    def average(a, b):
        valid = a[2] & b[2]
        return (np.where(valid[:, None], (a[0] + b[0]) * .5, 0),
                np.minimum(a[1], b[1]) * valid, valid)

    for side in ('kick', 'support'):
        left = left_kick if side == 'kick' else ~left_kick
        for name, l, r in (('shoulder', 5, 6), ('hip', 11, 12), ('knee', 13, 14),
                           ('ankle', 15, 16), ('big_toe', 17, 20),
                           ('small_toe', 18, 21), ('heel', 19, 22)):
            points[side + '_' + name] = point(np.where(left, l, r), known)
        points[side + '_toes'] = average(points[side + '_big_toe'], points[side + '_small_toe'])
    for name, a, b in (('hip_mid', 11, 12), ('shoulder_mid', 5, 6)):
        points[name] = average(point(np.full(len(values), a), np.ones(len(values), bool)),
                               point(np.full(len(values), b), np.ones(len(values), bool)))

    columns, names, vectors = [], [], {}
    for name, start, end in VECTORS:
        a, b = points[start], points[end]
        vector = b[0] - a[0]; length = np.linalg.norm(vector, axis=1)
        valid = a[2] & b[2] & (length > 1e-4)
        vector = np.where(valid[:, None], vector, 0)
        unit = vector / np.maximum(length, 1e-4)[:, None]
        confidence = np.minimum(a[1], b[1]) * valid
        columns.append(np.c_[vector, length * valid, unit, confidence, valid])
        names.extend(name + '_' + k for k in ('dx', 'dy', 'length', 'ux', 'uy', 'confidence', 'valid'))
        vectors[name] = unit, confidence, valid
    for first, second in ANGLES:
        a, b = vectors[first], vectors[second]; valid = a[2] & b[2]
        cosine = np.sum(a[0] * b[0], 1) * valid
        sine = (a[0][:, 0] * b[0][:, 1] - a[0][:, 1] * b[0][:, 0]) * valid
        columns.append(np.c_[cosine, sine, np.minimum(a[1], b[1]) * valid, valid])
        names.extend(first + '_to_' + second + '_' + k for k in ('cos', 'sin', 'confidence', 'valid'))
    result = np.concatenate(columns, 1).astype(np.float32)
    assert result.shape == (len(values), 117) and np.isfinite(result).all()
    return result, names


def reference_predictions(directory, rows, labels, indices, seed, stage):
    """Read held-out control predictions only; verify clip/label/class alignment."""
    path = Path(directory) / stage / 'mirror_control' / str(seed) / 'predictions.json'
    saved = json.loads(path.read_text())
    assert [p['clip'] for p in saved] == rows.iloc[indices]['clip'].tolist()
    np.testing.assert_array_equal([p['truth'] for p in saved], labels[indices])
    result = [np.array([p['probabilities'][k] for p in saved]) for k in range(3)]
    for k, count in enumerate((3, 3, 2)):
        assert result[k].shape == (len(indices), count) and np.isfinite(result[k]).all()
        assert (result[k] >= 0).all()
        np.testing.assert_allclose(result[k].sum(1), 1, atol=1e-5)
        np.testing.assert_array_equal(result[k].argmax(1), [p['prediction'][k] for p in saved])
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def execute(archive, folds, reference, out, budget=600):
    """One-factor kicker-only feature test; reuse verified control heads verbatim."""
    import torch
    import xgboost
    import pose_tree_controls as controls
    import pose_mirror
    import rgb_gru as model
    import rgb_gru_data as data
    import sparse_role_comparison as sparse
    assert torch.cuda.is_available(), 'Kaggle CUDA required'
    assert xgboost.__version__ == '3.4.1'
    out, reference = Path(out), Path(reference); out.mkdir(parents=True, exist_ok=True)
    checks = json.loads((reference / 'artifact_manifest.json').read_text())
    for name, digest in checks.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert data.sha(reference / name) == digest, name
    rows, arrays, labels, margins, ix = controls.load_inputs(archive, folds)
    n = len(rows); seeds = (6643, 6644, 6645); started = time.monotonic()
    old_protocol = json.loads((reference / 'protocol.json').read_text())
    assert old_protocol['folds_sha256'] == data.sha(folds) and old_protocol['archive_sha256'] == data.sha(archive)
    assert old_protocol['seeds'] == list(seeds) and old_protocol['xgb_rounds'] == [45, 19]
    assert old_protocol['xgboost'] == xgboost.__version__
    protocol = {'source_sha256': data.sha(__file__), 'helper_sha256': data.sha(pose_mirror.__file__),
                'input_helper_sha256': data.sha(controls.__file__), 'archive_sha256': data.sha(archive),
                'folds_sha256': data.sha(folds), 'reference_manifest_sha256': data.sha(reference / 'artifact_manifest.json'),
                'seeds': list(seeds), 'rounds': 45, 'depth': 4, 'inner_gate_delta': .005,
                'change': 'Append 117 explicit 2D kicker relations to 189 original features; refit kicker only.',
                'unchanged_heads': 'Keeper and outcome probabilities copied verbatim from matched R16 mirror control.',
                'selection': 'Three-seed inner mean, one fixed feature recipe; outer only if gain >= .005.',
                'sources': ['https://pubmed.ncbi.nlm.nih.gov/37409691/',
                            'https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.818556/full'],
                'limitation': '2D projected body relations, not true 3D angles; repeated exploratory folds.'}
    data.immutable_json(out / 'protocol.json', protocol)
    data.dump(out / 'hardware.json', {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                                    'platform': 'Kaggle', 'xgboost': xgboost.__version__})
    result = {'status': 'running'}; stored = {}
    for stage in ('inner', 'outer'):
        if stage == 'outer' and not result['promoted']: break
        x, y, m, fit, weights, _ = controls.make_variant(arrays[stage], labels, margins[stage], rows.source_id,
                                                       ix[stage + '_fit'], 'mirror_control')
        extra, names = relational_features(x); values = np.c_[x, extra].astype(np.float32)
        data.dump(out / (stage + '_feature_manifest.json'), {'shape': list(values.shape),
            'sha256': hashlib.sha256(values.tobytes()).hexdigest(), 'derived_columns': names,
            'missing_values': 'Zero plus explicit confidence/validity; unknown kicking foot masks semantic leg features.'})
        trials, baseline = [], []; check = ix[stage + '_check']; stored[stage] = []
        for seed in seeds:
            if time.monotonic() - started > budget - 60:
                raise TimeoutError('Safe fit boundary; completed models retained')
            original, original_sha = reference_predictions(reference, rows, labels, check, seed, stage)
            directory = out / stage / str(seed)
            p, info = pose_mirror.fit(values, y[:, 0], m[:, :3], fit, None, weights, 3, 4, 45, seed, directory)
            probabilities = [p[check], original[1], original[2]]
            prediction = np.column_stack([a.argmax(1) for a in probabilities])
            control = np.column_stack([a.argmax(1) for a in original])
            metrics = model.metric_report(labels[check], prediction)
            control_metrics = model.metric_report(labels[check], control)
            data.dump(directory / 'predictions.json', [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                'prediction': prediction[j].tolist(), 'probabilities': [a[j].tolist() for a in probabilities]}
                for j, i in enumerate(check)])
            trial = {'seed': seed, **metrics, 'fit': info, 'reference_prediction_sha256': original_sha}
            data.dump(directory / 'result_summary.json', trial)
            trials.append(trial); baseline.append(control_metrics)
            stored[stage].append(((prediction == labels[check]).astype(float) - (control == labels[check]).astype(float)))
            print(json.dumps({'stage': stage, 'seed': seed, 'weighted_score': metrics['weighted_score'],
                              'control_score': control_metrics['weighted_score']}), flush=True)
        score = float(np.mean([t['weighted_score'] for t in trials]))
        control_score = float(np.mean([t['weighted_score'] for t in baseline]))
        result[stage] = {'trials': trials, 'mean': score, 'control_mean': control_score, 'delta': score - control_score,
                         'paired': sparse.source_interval(np.mean(stored[stage], 0) @ np.asarray(data.WEIGHTS), rows.iloc[check].source_id)}
        if stage == 'inner': result['promoted'] = score >= control_score + .005
        data.dump(out / 'summary.json', result)
    result.update(status='complete', seconds=time.monotonic() - started, goal_target=.8,
                  note='Exploratory source-fold validation, not Kaggle competition public score.')
    data.dump(out / 'summary.json', result)
    return result
