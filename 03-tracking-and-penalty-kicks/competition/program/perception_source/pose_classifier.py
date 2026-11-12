"""R3 controlled body/foot/view/time ablations on sparse automatic poses."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import structured_probe as learner

VARIANTS = ('body_last', 'body_foot_last', 'body_foot_view', 'pose_motion_view')


def normalized_pose(points, scores, mask):
    """Hip-centered crop pose with torso scale, confidence and visibility flags."""
    points, scores, mask = np.asarray(points), np.asarray(scores), np.asarray(mask)
    valid = (scores > .3) & mask[..., None] & np.isfinite(points).all(-1) & np.isfinite(scores)
    result = np.zeros((*scores.shape, 4), np.float32)
    for t in range(len(points)):
        for role in range(2):
            p, v = points[t, role], valid[t, role]
            if not v.any(): continue
            hip = p[[11, 12]].mean(0) if v[[11, 12]].all() else None
            shoulder = p[[5, 6]].mean(0) if v[[5, 6]].all() else None
            body_valid = v.copy(); body_valid[17:] = False
            fallback_center = p[body_valid].mean(0) if body_valid.any() else np.array([112., 112.])
            center = hip if hip is not None else (shoulder if shoulder is not None else fallback_center)
            scale = np.linalg.norm(hip - shoulder) if hip is not None and shoulder is not None else 0.
            if scale < 5: scale = max(5., np.ptp(p[body_valid, 1]) / 2) if body_valid.any() else 112.
            result[t, role, v, :2] = np.clip((p[v] - center) / scale, -5, 5)
            result[t, role, v, 2] = np.clip(scores[t, role, v], 0, 1)
            result[t, role, v, 3] = 1.
    return result


def representations(points, scores, mask, timestamps, view):
    pose = normalized_pose(points, scores, mask)
    static = pose[-1].flatten()
    pair = pose[1:, :, :, 3] * pose[:-1, :, :, 3]
    dt = np.diff(timestamps)
    assert (dt > 0).all()
    delta = np.clip(np.diff(pose[..., :2], axis=0) / dt[:, None, None, None], -20, 20) * pair[..., None]
    return {'body_last': pose[-1, :, :17].flatten(), 'body_foot_last': static,
            'body_foot_view': np.r_[static, view[-1]],
            'pose_motion_view': np.r_[pose.flatten(), delta.flatten(), pair.flatten(), view.flatten()]}


def build_features(pose_dir, image_bundle, geometry_bundle, out):
    started = time.monotonic()
    manifest = json.loads((image_bundle / 'manifest.json').read_text())
    rows = learner.pd.read_csv(image_bundle / 'public_folds.csv')
    geo = np.load(geometry_bundle / 'geometry.npz')
    gm = json.loads((geometry_bundle / 'manifest.json').read_text())
    assert learner.data.sha(geometry_bundle / 'geometry.npz') == gm['files']['geometry.npz']
    np.testing.assert_array_equal(rows['clip'], geo['clips'])
    assert len(rows) == 1299 and rows.groupby('source_id').fold.nunique().max() == 1
    values, hashes, coverage = [], {}, []
    for i, original in enumerate(manifest['records']):
        clip = original['clip']; assert clip == rows.iloc[i]['clip']
        record = json.loads((pose_dir / 'records' / (clip + '.json')).read_text())
        assert record['input'] == original
        path = pose_dir / 'features' / (clip + '.npz')
        assert learner.data.sha(path) == record['sha256']; hashes[clip] = record['sha256']
        with np.load(path) as a:
            relative = geo['relative_motion'][i, :48].reshape(3, 16)
            quality = geo['quality_only'][i].reshape(3, 4)
            view = np.c_[relative[:, :2], quality[:, 0]]
            values.append(representations(a['points'], a['scores'], a['mask'], a['timestamps'], view))
            coverage.append((a['scores'] > .3).mean(axis=(0, 1)).tolist())
    arrays = {v: torch.from_numpy(np.stack([x[v] for x in values]).astype(np.float32)) for v in VARIANTS}
    assert all(torch.isfinite(x).all() for x in arrays.values())
    learner.data.dump(out / 'feature_manifest.json', {'pose_hashes': hashes,
                      'geometry_sha256': gm['files']['geometry.npz'],
                      'pose_protocol_sha256': learner.data.sha(pose_dir / 'protocol.json'),
                      'dimensions': {v: x.shape[1] for v, x in arrays.items()},
                      'mean_joint_available_fraction': np.mean(coverage, axis=0).tolist(),
                      'seconds': time.monotonic() - started,
                      'note': 'Confidence/availability is not pose accuracy; view is approximate goal-box/ball geometry.'})
    np.savez_compressed(out / 'features.npz', **{v: x.numpy() for v, x in arrays.items()}, clips=rows['clip'].to_numpy(dtype=str))
    return rows, arrays


def execute(pose_dir, image_bundle, geometry_bundle, out, budget=600, seed=6643):
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); deadline = started + budget - 60
    rows, arrays = build_features(pose_dir, image_bundle, geometry_bundle, out)
    meta = torch.nn.functional.one_hot(torch.tensor([{'left': 0, 'right': 1}.get(f, 2) for f in rows.preferred_foot]), 3).float()
    labels = torch.tensor([[cs.index(row[k]) for k, cs in learner.data.TARGETS.items()] for _, row in rows.iterrows()])
    ix = lambda a: torch.tensor(np.flatnonzero(a))
    train, val, inner, check = ix(rows.fold != 0), ix(rows.fold == 0), ix(~rows.fold.isin([0, 1])), ix(rows.fold == 1)
    assert set(rows.iloc[train].source_id).isdisjoint(rows.iloc[val].source_id)
    assert set(rows.iloc[inner].source_id).isdisjoint(rows.iloc[check].source_id)
    protocol = {'source_sha256': learner.data.sha(__file__), 'learner_sha256': learner.data.sha(learner.__file__),
                'feature_sha256': learner.data.sha(out / 'features.npz'),
                'folds_sha256': learner.data.sha(image_bundle / 'public_folds.csv'),
                'seed': seed, 'variants': VARIANTS, 'configs': learner.CONFIGS,
                'outer_fold': 0, 'inner_fold': 1, 'max_epochs': 200, 'learning_rate': .003,
                'gpu': torch.cuda.get_device_name(0)}
    learner.data.immutable_json(out / 'protocol.json', json.loads(json.dumps(protocol)))
    baseline, base_pred, _ = learner.research.predict_metrics(learner.research.foot_priors(meta, labels, train), labels, val)
    results = {'foot_rule': {'metrics': baseline}}
    try:
        for variant, x in arrays.items():
            directory = out / variant; candidates = []
            for config in learner.CONFIGS:
                _, info = learner.fit(x, meta, labels, inner, check, config, 200,
                                      directory / ('inner_' + config['kind']), deadline, seed)
                candidates.append(info)
            best = max(candidates, key=lambda c: (c['inner_score'], -c['parameters']))
            logits, info = learner.fit(x, meta, labels, train, None, best['config'], best['best_epoch'],
                                       directory / 'outer', deadline, seed)
            metrics, pred, probs = learner.research.predict_metrics(logits, labels, val)
            delta = ((pred[val] == labels[val]).float() - (base_pred[val] == labels[val]).float()) @ torch.tensor(learner.data.WEIGHTS)
            result = {'metrics': metrics, 'candidates': candidates, **info,
                      'vs_foot': learner.sparse.source_interval(delta.numpy(), rows.iloc[val].source_id)}
            records = [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(), 'prediction': pred[i].tolist(),
                        'probabilities': [p[i].tolist() for p in probs]} for i in val]
            learner.data.dump(directory / 'predictions.json', records)
            learner.data.dump(directory / 'result.json', result); results[variant] = result
            print(json.dumps({'stage': variant, 'score': metrics['weighted_score']}), flush=True)
        learner.data.dump(out / 'summary.json', results)
        learner.data.dump(out / 'status.json', {'stage': 'complete', 'seconds': time.monotonic() - started})
    except Exception as exc:
        learner.data.dump(out / 'status.json', {'stage': 'error', 'error': str(exc)}); raise
    finally:
        learner.data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): learner.data.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
