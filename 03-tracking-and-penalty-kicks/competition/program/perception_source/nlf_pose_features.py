"""R18 cached frozen NLF joints with controlled 3D representations."""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import nlf_pose_pilot as pilot

JOINT_SWAP = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15,
                       17, 16, 19, 18, 21, 20, 23, 22])


def reflect(points, uncertainty):
    return np.asarray(points)[JOINT_SWAP] * [-1., 1., 1.], np.asarray(uncertainty)[JOINT_SWAP]


def geometry(points, uncertainty, agreement, basis):
    """123 features; projection/uncertainty gates are quality proxies only."""
    assert basis in ('camera', 'body')
    p, u = np.asarray(points, float), np.asarray(uncertainty, float)
    assert p.shape == (24, 3) and u.shape == (24,)
    result = np.zeros(123, np.float32)
    if not np.isfinite(p).all() or not np.isfinite(u).all() or agreement is None: return result, False
    visible = (u < .25) & (u >= 0)  # Official NLF plausibility threshold, in metres.
    hip = (p[1] + p[2]) * .5; shoulder = (p[16] + p[17]) * .5
    torso = float(np.linalg.norm(shoulder - hip))
    eligible = (0 <= agreement <= .12 and 200 <= torso <= 900 and
                visible.mean() > 1/3 and visible[[0, 1, 2, 16, 17]].all())
    result[-3:] = [float(eligible), np.clip(agreement, 0, 1), visible.mean()]
    if not eligible: return result, False
    coords = (p - p[0]) / torso
    if basis == 'body':
        x = p[2] - p[1]
        if np.linalg.norm(x) < .1 * torso:
            result[-3] = 0; return result, False
        x /= np.linalg.norm(x)
        y = shoulder - hip; y = y - np.dot(y, x) * x
        if np.linalg.norm(y) < .1 * torso:
            result[-3] = 0; return result, False
        y /= np.linalg.norm(y); z = np.cross(x, y)
        coords = coords @ np.stack((x, y, z), axis=1)
    joints = np.c_[np.clip(coords, -5, 5), np.clip(u * 1000 / torso, 0, 5), visible]
    joints[~visible] = 0
    result[:120] = joints.flatten()
    assert np.isfinite(result).all()
    return result, True


def prepare(bundle, archive, assets, out):
    """Read matched selections and poses only; no perception model executes."""
    bundle, archive, assets, out = map(Path, (bundle, archive, assets, out))
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text())
    asset_manifest = json.loads((assets / 'nlf-assets.json').read_text())
    for name, digest in asset_manifest['files'].items(): assert pilot.sha(assets / name) == digest
    assert pilot.sha(bundle / 'images.bin') == manifest['files']['images.zip']
    assert pilot.sha(bundle / 'public_folds.csv') == manifest['files']['public_folds.csv']
    rows = pd.read_csv(bundle / 'public_folds.csv')
    with np.load(assets / 'selection.npz') as z: selection = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(rows['clip'], selection['clips'])
    np.testing.assert_array_equal(rows.fold, selection['folds'])
    assert len(rows) == 1299 and rows.groupby('source_id').fold.nunique().max() == 1
    assert '0526_public.mp4' not in set(rows['clip'])
    assert pilot.sha(archive) == 'dd675833d1c43a1d63559cfcb0e85c2e857d8ea4d42bfd0a115d791430e63831'
    with zipfile.ZipFile(archive) as z:
        raw = z.read('results/r10_pose_matched/poses.npz')
        state = json.loads(z.read('results/r10_pose_matched/state.json'))
        assert hashlib.sha256(raw).hexdigest() == state['poses_sha256']
        with np.load(io.BytesIO(raw)) as a:
            poses = {h: (p, s) for h, p, s in zip(a['hashes'].tolist(), a['points'], a['scores'])}
    keys = {h: k for k, h in manifest['image_hashes'].items()}
    out.mkdir(parents=True, exist_ok=True)
    inference = {'model_sha256': pilot.MODEL_SHA, 'fov': 55., 'num_aug': 1, 'internal_batch_size': 4,
                 'input': 'Existing letterbox224 RGB JPEG', 'box': 'RTMW visible joints + 15% margin, min4 pixels',
                 'weights': 'Last24 canonical smpl joints, no parametric fitting or detection',
                 'raw_units': {'points3d': 'millimetres', 'uncertainty': 'metres'}}
    protocol = out / 'inference_protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text()) == inference
    else: pilot.dump(protocol, inference)
    # Import the already computed 12 pilot poses without model execution.
    with zipfile.ZipFile(assets / 'pilot-cache.bin') as z:
        checks = json.loads(z.read('artifact_manifest.json'))
        for name, digest in checks.items(): assert hashlib.sha256(z.read(name)).hexdigest() == digest
        old = json.loads(z.read('protocol.json'))
        assert old['model_sha256'] == pilot.MODEL_SHA and old['default_crop_fov_degrees'] == 55.
        assert old['num_aug'] == 1 and old['internal_batch_size'] == 4
        for r in json.loads(z.read('records.json')):
            h = r['image_sha256']; target = out / 'poses' / (h + '.npz'); record = out / 'records' / (h + '.json')
            if record.exists(): continue
            blob = z.read(r['clip'] + '.npz'); assert hashlib.sha256(blob).hexdigest() == r['output_sha256']
            np.testing.assert_allclose(pilot.observed_box(*poses[h]), r['box'], atol=1e-5)
            target.parent.mkdir(exist_ok=True); target.write_bytes(blob)
            pilot.dump(record, {'input_sha256': h, 'sha256': pilot.sha(target), 'inference_protocol_sha256': pilot.sha(protocol),
                               'agreement': r['projected_joint_median_error_fraction'], 'pilot_reused': True, 'usable_box': True})
    return rows, selection, poses, keys


def extract(rows, selection, poses, keys, image_archive, assets, out, stage, budget=600):
    import torch
    import torchvision
    from PIL import Image
    assert torch.cuda.is_available() and stage in ('inner', 'outer')
    out, assets = Path(out), Path(assets); torch.set_num_threads(4)
    ix = np.flatnonzero(rows.fold != 0) if stage == 'inner' else np.arange(len(rows))
    wanted = sorted(set(selection[stage + '_hashes'][ix, 0]) - {''})
    protocol_sha = pilot.sha(out / 'inference_protocol.json')
    pending = []
    for h in wanted:
        record = out / 'records' / (h + '.json')
        if record.exists():
            r = json.loads(record.read_text()); assert r['input_sha256'] == h and r['inference_protocol_sha256'] == protocol_sha
            assert pilot.sha(out / 'poses' / (h + '.npz')) == r['sha256']
        else: pending.append(h)
    started = time.monotonic(); completed = len(wanted) - len(pending); reused = completed; net = None
    if pending:
        path = assets / 'nlf_l_multi_0.3.2.patch6.torchscript'; assert pilot.sha(path) == pilot.MODEL_SHA
        net = torch.jit.load(str(path), map_location='cuda').eval()
        with torch.inference_mode(), torch.device('cuda'):
            field = net.get_weights_for_canonical_points(net.cano_all['smpl'][-24:].cuda().float())
    with zipfile.ZipFile(image_archive) as z:
        for h in pending:
            if time.monotonic() - started > budget - 60: break
            blob = z.read(keys[h]); assert hashlib.sha256(blob).hexdigest() == h
            points, scores = poses[h]; box = pilot.observed_box(points, scores)
            a = np.zeros((24, 3), np.float32); b = np.zeros((24, 2), np.float32); u = np.ones(24, np.float32)
            error = None
            if box is not None:
                pixels = np.asarray(Image.open(io.BytesIO(blob)).convert('RGB')).copy()
                assert pixels.shape == (224, 224, 3)
                image = torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda()
                with torch.inference_mode(), torch.device('cuda'):
                    pred = net.estimate_poses_batched(image, [torch.from_numpy(box)[None].cuda()], field,
                        default_fov_degrees=55., num_aug=1, internal_batch_size=4)
                assert pred['poses3d'][0].is_cuda
                a, b, u = [pred[k][0][0].cpu().numpy() for k in ('poses3d', 'poses2d', 'uncertainties')]
                assert a.shape == (24, 3) and b.shape == (24, 2) and u.shape == (24,)
                error = pilot.projection_error(b, points, scores, box[3])
                if error is not None and not np.isfinite(error): error = None
            target = out / 'poses' / (h + '.npz'); target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as f: np.savez_compressed(f, points3d=a, points2d=b, uncertainty=u, image_sha256=h)
            temporary.replace(target)
            pilot.dump(out / 'records' / (h + '.json'), {'input_sha256': h, 'sha256': pilot.sha(target),
                       'inference_protocol_sha256': protocol_sha, 'agreement': error, 'pilot_reused': False, 'usable_box': box is not None})
            completed += 1
            if completed % 100 == 0: print(json.dumps({'stage': stage + '_nlf', 'completed': completed, 'total': len(wanted), 'reused': reused}), flush=True)
    report = {'status': 'complete' if completed == len(wanted) else 'budget_boundary', 'stage': stage,
              'completed': completed, 'wanted': len(wanted), 'reused': reused, 'new_inferences': completed-reused,
              'seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(0)}
    pilot.dump(out / (stage + '_extraction.json'), report); del net; torch.cuda.empty_cache()
    return report


def feature_array(selection, cache, stage, basis):
    n = len(selection['clips']); result = np.zeros((2 * n, 123), np.float32); valid = np.zeros(2*n, bool)
    for i, h in enumerate(selection[stage + '_hashes'][:, 0]):
        if not h or (stage == 'inner' and selection['folds'][i] == 0): continue
        cache = Path(cache); record = json.loads((cache / 'records' / (h + '.json')).read_text())
        path = cache / 'poses' / (h + '.npz'); assert pilot.sha(path) == record['sha256']
        with np.load(path) as a: points, uncertainty = a['points3d'], a['uncertainty']
        result[i], valid[i] = geometry(points, uncertainty, record['agreement'], basis)
        result[i+n], valid[i+n] = geometry(*reflect(points, uncertainty), record['agreement'], basis)
    return result, valid


def execute(bundle, archive, assets, reference, out, budget=1200):
    import torch
    import xgboost
    import pose_tree_controls as controls
    import pose_mirror
    import pose_biomechanics as references
    import rgb_gru as model
    import rgb_gru_data as data
    import sparse_role_comparison as sparse
    assert torch.cuda.is_available() and xgboost.__version__ == '3.4.1'
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cache = out / 'features'; started = time.monotonic(); deadline = started + budget - 90
    rows, selection, poses, keys = prepare(bundle, archive, assets, cache)
    baseline_rows, arrays, labels, margins, ix = controls.load_inputs(archive, Path(bundle) / 'public_folds.csv')
    assert rows['clip'].tolist() == baseline_rows['clip'].tolist()
    np.testing.assert_array_equal(labels, selection['labels'])
    for stage in ('inner', 'outer'): np.testing.assert_array_equal(arrays[stage], selection[stage + '_pose'])
    reference = Path(reference); checks = json.loads((reference / 'artifact_manifest.json').read_text())
    for name, digest in checks.items(): assert data.sha(reference / name) == digest
    seeds = (6643, 6644, 6645); variants = ('camera', 'body'); n = len(rows)
    protocol = {'sources': {Path(p).name: data.sha(p) for p in (__file__, pilot.__file__, controls.__file__, pose_mirror.__file__)},
                'selection_sha256': data.sha(Path(assets) / 'selection.npz'), 'model_sha256': pilot.MODEL_SHA,
                'input_archive_sha256': data.sha(archive), 'reference_manifest_sha256': data.sha(reference / 'artifact_manifest.json'),
                'seeds': list(seeds), 'variants': list(variants), 'depth': 4, 'rounds': 45,
                'quality': 'Agreement<=.12, 200<=torso_mm<=900, >1/3 joints uncertainty<.25m, pelvis/hips/shoulders valid; body frame nondegenerate.',
                'fallback': 'Original matched seed control probabilities for unusable 3D; retain every held-out clip.',
                'selection': 'Best three-seed inner mean >= matched control+.005; outer refit only for that representation.',
                'unchanged_heads': 'Keeper/outcome copied verbatim from R16 control; all fit rows retain original/mirror mass .5.',
                'geometry': 'Uncalibrated crop FOV55; 3D human-pose estimate, not field coordinates.'}
    data.immutable_json(out / 'protocol.json', protocol)
    pilot.dump(out / 'hardware.json', {'platform': 'Kaggle', 'gpu': torch.cuda.get_device_name(0),
                                     'torch': str(torch.__version__), 'xgboost': xgboost.__version__})
    report = {'status': 'running', 'inner': {}, 'outer': {}}

    def remaining(cap):
        value = int(deadline-time.monotonic())
        if value < 120: raise TimeoutError('Safe fit/extraction boundary; resume cached artifacts')
        return min(cap, value)

    def stage_features(stage):
        status = extract(rows, selection, poses, keys, Path(bundle) / 'images.bin', assets, cache, stage, remaining(600))
        print(json.dumps({'stage': 'extraction_complete', **status}), flush=True)
        if status['status'] != 'complete': raise TimeoutError('Partial NLF features preserved')

    def trial_group(stage, variant):
        extra, eligible = feature_array(selection, cache, stage, variant)
        base, y, margin, fit, weights, _ = controls.make_variant(arrays[stage], labels, margins[stage], rows.source_id,
                                                               ix[stage + '_fit'], 'mirror_control')
        x = np.c_[base, extra].astype(np.float32); val = ix[stage + '_check']; trials = []; deltas = []; control_scores = []
        pilot.dump(out / (stage + '_' + variant + '_features.json'), {'shape': list(x.shape), 'sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                    'eligible_total': int(eligible[:n].sum()), 'eligible_validation': int(eligible[val].sum()), 'validation_count': len(val)})
        for seed in seeds:
            remaining(120)
            original, original_sha = references.reference_predictions(reference, rows, labels, val, seed, stage)
            directory = out / stage / variant / str(seed)
            prediction, info = pose_mirror.fit(x, y[:, 0], margin[:, :3], fit, None, weights, 3, 4, 45, seed, directory)
            kick = prediction[val].copy(); kick[~eligible[val]] = original[0][~eligible[val]]
            probabilities = [kick, original[1], original[2]]
            pred = np.column_stack([p.argmax(1) for p in probabilities]); control = np.column_stack([p.argmax(1) for p in original])
            metric = model.metric_report(labels[val], pred); control_scores.append(model.metric_report(labels[val], control)['weighted_score'])
            pilot.dump(directory / 'predictions.json', [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                'prediction': pred[j].tolist(), 'probabilities': [p[j].tolist() for p in probabilities], 'used_3d': bool(eligible[i])}
                for j, i in enumerate(val)])
            result = {'seed': seed, **metric, 'fit': info, 'reference_prediction_sha256': original_sha}
            pilot.dump(directory / 'result_summary.json', result); trials.append(result)
            deltas.append((pred == labels[val]).astype(float) - (control == labels[val]).astype(float))
            print(json.dumps({'stage': stage, 'variant': variant, 'seed': seed, 'score': metric['weighted_score'],
                              'used_3d': int(eligible[val].sum()), 'fallback': int((~eligible[val]).sum())}), flush=True)
        return {'trials': trials, 'mean': float(np.mean([t['weighted_score'] for t in trials])),
                'control_mean': float(np.mean(control_scores)),
                'paired': sparse.source_interval(np.mean(deltas, 0) @ np.array(data.WEIGHTS), rows.iloc[val].source_id)}

    stage_features('inner')
    for variant in variants:
        report['inner'][variant] = trial_group('inner', variant); pilot.dump(out / 'summary.json', report)
    chosen = max(variants, key=lambda v: report['inner'][v]['mean'])
    promoted = chosen if report['inner'][chosen]['mean'] >= report['inner'][chosen]['control_mean'] + .005 else None
    report['promoted'] = promoted; pilot.dump(out / 'summary.json', report)
    if promoted:
        stage_features('outer'); report['outer'][promoted] = trial_group('outer', promoted)
    report.update(status='complete', seconds=time.monotonic()-started, goal_target=.8,
                  note='Exploratory source-fold validation only, not Kaggle competition public score.')
    pilot.dump(out / 'summary.json', report)
    return report
