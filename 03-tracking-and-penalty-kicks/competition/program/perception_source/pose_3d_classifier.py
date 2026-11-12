"""R19: matched CatBoost 2D/3D control on existing NLF features only."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import nlf_pose_features as nlf
import pose_tree_controls as controls
import pose_biomechanics as references
import rgb_gru_data as data


def integer_score(truth, prediction):
    """Exact numerator/denominator; avoid floating-point tie selection."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    assert truth.shape == prediction.shape and truth.ndim == 2 and truth.shape[1] == 3
    return int((truth == prediction).astype(int).sum(0) @ np.array([60, 25, 15])), 100 * len(truth)


def passes_gate(candidate, controls_, denominator):
    assert isinstance(candidate, int) and isinstance(denominator, int) and denominator > 0
    return 200 * (candidate - max(controls_)) >= denominator


def execute(archive, folds, cached_nlf, selection_file, reference, out, budget=600):
    import torch
    import catboost
    import rgb_gru as model
    import sparse_role_comparison as sparse
    assert torch.cuda.is_available() and catboost.__version__ == '1.2.10'
    out, reference, cached_nlf = map(Path, (out, reference, cached_nlf))
    out.mkdir(parents=True, exist_ok=True)
    for root in (reference, cached_nlf):
        checks = json.loads((root / 'artifact_manifest.json').read_text())
        for name, digest in checks.items(): assert data.sha(root / name) == digest
    old = json.loads((cached_nlf / 'summary.json').read_text())
    assert old['status'] == 'complete' and old['promoted'] == 'body'
    rows, arrays, labels, margins, ix = controls.load_inputs(archive, folds)
    with np.load(selection_file) as z: selection = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(rows['clip'], selection['clips']); np.testing.assert_array_equal(labels, selection['labels'])
    for stage in ('inner', 'outer'): np.testing.assert_array_equal(arrays[stage], selection[stage + '_pose'])
    seeds = (6643, 6644, 6645); n = len(rows); started = time.monotonic()
    protocol = {'source_sha256': data.sha(__file__), 'feature_source_sha256': data.sha(nlf.__file__),
        'classifier_helper_sha256': data.sha(controls.__file__), 'cached_nlf_manifest_sha256': data.sha(cached_nlf / 'artifact_manifest.json'),
        'reference_manifest_sha256': data.sha(reference / 'artifact_manifest.json'), 'selection_sha256': data.sha(selection_file),
        'seeds': list(seeds), 'catboost': catboost.__version__, 'trees': 450, 'depth': 4, 'learning_rate': .035, 'l2': 15,
        'variants': ['2d_control', '2d_plus_body3d'], 'unchanged_heads': 'R16 mirror-control keeper and outcome',
        'fallback': 'Matched CatBoost 2D kicker probability when NLF quality fails',
        'gate': 'Exact integer three-seed inner score >= both CatBoost 2D control and R18 body-XGBoost + .005; outer refit both arms only if passed.',
        'scope': 'One fixed comparison on exploratory source folds. No feature/model inference or outer-based branch selection.'}
    data.immutable_json(out / 'protocol.json', protocol)
    data.dump(out / 'hardware.json', {'platform': 'Kaggle', 'gpu': torch.cuda.get_device_name(0),
                                    'catboost_training': 'GPU', 'catboost_inference': 'CPU', 'catboost': catboost.__version__})
    report = {'status': 'running', 'inner': {}, 'outer': {}}
    for stage in ('inner', 'outer'):
        if stage == 'outer' and not report['promoted']: break
        base, y, margin, fit, weights, _ = controls.make_variant(arrays[stage], labels, margins[stage], rows.source_id,
                                                               ix[stage + '_fit'], 'mirror_control')
        extra, valid = nlf.feature_array(selection, cached_nlf / 'features', stage, 'body')
        full = np.c_[base, extra].astype(np.float32); check = ix[stage + '_check']
        expected = json.loads((cached_nlf / (stage + '_body_features.json')).read_text())
        assert hashlib.sha256(full.tobytes()).hexdigest() == expected['sha256'], 'Frozen feature reproduction required'
        trials = {'2d_control': [], '2d_plus_body3d': []}; scores = {k: 0 for k in trials}; paired = []; versus_old = []
        old_numerator = 0
        for seed in seeds:
            if time.monotonic()-started > budget-60: raise TimeoutError('Safe classifier boundary; models saved')
            original, original_sha = references.reference_predictions(reference, rows, labels, check, seed, stage)
            if stage == 'inner':
                path = reference / stage / 'catboost_mirror' / str(seed)
                config = json.loads((path / 'target0/protocol.json').read_text())
                assert config['x_sha256'] == hashlib.sha256(base.tobytes()).hexdigest()
                assert config['margin_sha256'] == hashlib.sha256(margin[:, :3].tobytes()).hexdigest()
                assert config['fit_indices'] == fit.tolist() and config['weights'] == weights.tolist()
                assert config['catboost'] == catboost.__version__
                saved = json.loads((path / 'predictions.json').read_text())
                assert [r['clip'] for r in saved] == rows.iloc[check]['clip'].tolist()
                np.testing.assert_array_equal([r['truth'] for r in saved], labels[check])
                control = np.array([r['probabilities'][0] for r in saved])
                control_info = {'reused': True, 'predictions_sha256': data.sha(path / 'predictions.json'),
                                'model_sha256': data.sha(path / 'target0/model.cbm')}
            else:
                control_all, control_info = controls.fit_catboost(base, y[:, 0], margin[:, :3], fit, weights,
                    out / stage / '2d_control' / str(seed), seed)
                control = control_all[check]
            full_all, full_info = controls.fit_catboost(full, y[:, 0], margin[:, :3], fit, weights,
                out / stage / '2d_plus_body3d' / str(seed), seed)
            candidate = full_all[check].copy(); candidate[~valid[check]] = control[~valid[check]]
            predictions = {}
            for variant, kick, info in [('2d_control', control, control_info), ('2d_plus_body3d', candidate, full_info)]:
                probabilities = [kick, original[1], original[2]]
                prediction = np.column_stack([p.argmax(1) for p in probabilities]); predictions[variant] = prediction
                metric = model.metric_report(labels[check], prediction); numerator, denominator = integer_score(labels[check], prediction)
                scores[variant] += numerator
                directory = out / stage / variant / str(seed)
                data.dump(directory / 'predictions.json', [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                    'prediction': prediction[j].tolist(), 'probabilities': [p[j].tolist() for p in probabilities],
                    'used_3d': bool(valid[i]) if variant == '2d_plus_body3d' else False} for j, i in enumerate(check)])
                trial = {'seed': seed, **metric, 'fit': info, 'exact_numerator': numerator, 'exact_denominator': denominator,
                         'reference_predictions_sha256': original_sha}
                data.dump(directory / 'result_summary.json', trial); trials[variant].append(trial)
                print(json.dumps({'stage': stage, 'variant': variant, 'seed': seed, 'score': metric['weighted_score']}), flush=True)
            paired.append((predictions['2d_plus_body3d'] == labels[check]).astype(float) -
                          (predictions['2d_control'] == labels[check]).astype(float))
            old_records = json.loads((cached_nlf / stage / 'body' / str(seed) / 'predictions.json').read_text())
            assert [r['clip'] for r in old_records] == rows.iloc[check]['clip'].tolist()
            old_pred = np.array([r['prediction'] for r in old_records])
            old_numerator += integer_score(labels[check], old_pred)[0]
            versus_old.append((predictions['2d_plus_body3d'] == labels[check]).astype(float) - (old_pred == labels[check]).astype(float))
        total_denominator = denominator * len(seeds)
        report[stage] = {v: {'trials': trials[v], 'mean': scores[v] / total_denominator,
                            'exact_numerator': scores[v], 'exact_denominator': total_denominator} for v in trials}
        report[stage]['paired_vs_2d_catboost'] = sparse.source_interval(np.mean(paired, 0) @ np.asarray(data.WEIGHTS), rows.iloc[check].source_id)
        report[stage]['paired_vs_3d_xgboost'] = sparse.source_interval(np.mean(versus_old, 0) @ np.asarray(data.WEIGHTS), rows.iloc[check].source_id)
        if stage == 'inner':
            report['promoted'] = passes_gate(scores['2d_plus_body3d'], [scores['2d_control'], old_numerator], total_denominator)
            report['reference_3d_xgboost_exact_numerator'] = old_numerator
        data.dump(out / 'summary.json', report)
    report.update(status='complete', seconds=time.monotonic()-started, goal_target=.8,
                  note='Repeated exploratory source folds, not competition public score.')
    data.dump(out / 'summary.json', report)
    return report
