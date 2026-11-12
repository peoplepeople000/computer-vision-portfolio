"""R15: corrected-role frozen DINO tokens and learned pose/RGB fusion.

CPU paths only validate/assemble cached data. Model execution requires CUDA.
R1-R14 modules remain frozen; their helpers are fingerprinted in the protocol.
"""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import dino_probe
import pose_mirror
from pose_classifier import normalized_pose
from pose_selection import select_pose
import rgb_gru_data as data


def source_hashes():
    import pose_classifier, pose_selection
    return {Path(p).name: data.sha(p) for p in (
        __file__, dino_probe.__file__, pose_mirror.__file__, data.__file__,
        pose_classifier.__file__, pose_selection.__file__)}


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def latest_selection(record, image_hashes, poses):
    """Use the exact R10 availability rule; retain matching image identities."""
    points = np.zeros((3, 2, 23, 2), np.float32)
    scores = np.zeros((3, 2, 23), np.float32)
    mask = np.asarray(record['mask'], bool)
    for t, role in np.argwhere(mask):
        points[t, role], scores[t, role] = poses[image_hashes[record['keys'][t][role]]]
    selected, anchors = select_pose(normalized_pose(points, scores, mask), mask,
                                    record['timestamps'], True)
    hashes = ['' if t < 0 else image_hashes[record['keys'][int(t)][role]]
              for role, t in enumerate(anchors)]
    return selected, hashes, anchors


def prepare(bundle, pose_archive, out):
    """Validate existing public caches and save aligned inner/outer selections."""
    bundle, pose_archive, out = map(Path, (bundle, pose_archive, out))
    manifest_path = bundle / 'bundle-manifest.json'
    if not manifest_path.exists(): manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    images = bundle / 'images.bin'
    if not images.exists(): images = bundle / 'images.zip'
    assert data.sha(images) == manifest['files']['images.zip']
    assert data.sha(bundle / 'public_folds.csv') == manifest['files']['public_folds.csv']
    rows = pd.read_csv(bundle / 'public_folds.csv')
    assert len(rows) == 1299 and '0526_public.mp4' not in set(rows['clip'])
    assert rows.groupby('source_id').fold.nunique().max() == 1
    assert manifest['role_training_folds'] == {'inner': [2, 3, 4], 'outer': [1, 2, 3, 4]}
    with zipfile.ZipFile(pose_archive) as archive:
        transfer = json.loads(archive.read('transfer_manifest.json'))
        for name, digest in transfer.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        original = json.loads(archive.read('r10_matched_manifest.json'))
        for key in ('records', 'image_hashes', 'role_training_folds'):
            assert original[key] == manifest[key], key
        assert original['files']['public_folds.csv'] == manifest['files']['public_folds.csv']
        state = json.loads(archive.read('results/r10_pose_matched/state.json'))
        assert state['completed'] == state['target'] and state['scope'] == 'full'
        raw_pose = archive.read('results/r10_pose_matched/poses.npz')
        assert hashlib.sha256(raw_pose).hexdigest() == state['poses_sha256']
        with np.load(io.BytesIO(raw_pose)) as saved:
            poses = {key: (p, s) for key, p, s in zip(saved['hashes'].tolist(), saved['points'], saved['scores'])}
        raw_features = archive.read('results/r10_classification/features.npz')
        classifier_manifest = json.loads(archive.read('results/r10_classification/artifact_manifest.json'))
        assert hashlib.sha256(raw_features).hexdigest() == classifier_manifest['features.npz']
        with np.load(io.BytesIO(raw_features)) as cached:
            np.testing.assert_array_equal(cached['clips'], rows['clip'])
            expected = {stage: cached[stage + '__new_pose'] for stage in ('inner', 'outer')}
    metadata = np.eye(3, dtype=np.float32)[[{'left': 0, 'right': 1}.get(f, 2) for f in rows.preferred_foot]]
    arrays = {'clips': rows['clip'].to_numpy(dtype=str), 'folds': rows.fold.to_numpy(),
              'labels': np.array([[c.index(row[k]) for k, c in data.TARGETS.items()] for _, row in rows.iterrows()])}
    for stage in ('inner', 'outer'):
        selected, hashes, anchors = [], [], []
        for index, record in enumerate(manifest['records'][stage]):
            assert record['clip'] == rows.iloc[index]['clip']
            p, h, a = latest_selection(record, manifest['image_hashes'], poses)
            selected.append(np.r_[p, metadata[index]]); hashes.append(h); anchors.append(a)
        arrays[stage + '_pose'] = np.asarray(selected, np.float32)
        arrays[stage + '_hashes'] = np.asarray(hashes, dtype='U64')
        arrays[stage + '_anchors'] = np.asarray(anchors, np.int64)
        np.testing.assert_array_equal(arrays[stage + '_pose'], expected[stage])
    # Verify payloads, not merely the container hash, before any encoder run.
    needed = set(arrays['inner_hashes'].flat) | set(arrays['outer_hashes'].flat)
    needed.discard('')
    by_hash = {digest: key for key, digest in manifest['image_hashes'].items() if digest in needed}
    with zipfile.ZipFile(images) as archive:
        for digest, key in by_hash.items():
            assert hashlib.sha256(archive.read(key)).hexdigest() == digest, key
    protocol = {'sources': source_hashes(), 'image_manifest_sha256': data.sha(manifest_path),
                'pose_archive_sha256': data.sha(pose_archive), 'pose_sha256': state['poses_sha256'],
                'r10_features_sha256': classifier_manifest['features.npz'],
                'folds_sha256': data.sha(bundle / 'public_folds.csv'),
                'model': dino_probe.MODEL, 'revision': dino_probe.REVISION,
                'weights_sha256': dino_probe.WEIGHT_SHA,
                'tokens': 'CLS plus 2x2 average-pooled 16x16 patch grid = 65 tokens, 384 dimensions, float16',
                'mirror': 'Encode horizontally flipped pixels independently; anatomical pose and direction/foot labels swap.',
                'selection': 'Exact R10 latest observed pose per role; separate nested role selectors.'}
    data.immutable_json(out / 'selection_protocol.json', protocol)
    atomic_npz(out / 'selection.npz', **arrays)
    data.dump(out / 'selection_state.json', {'sha256': data.sha(out / 'selection.npz'), 'clips': len(rows),
              'unique_images': len(needed), 'exact_r10_pose_match': True, 'image_payloads_verified': len(by_hash)})
    return rows, arrays, images, by_hash


def spatial_tokens(hidden):
    assert hidden.ndim == 3 and hidden.shape[1:] == (257, 384)
    grid = hidden[:, 1:].transpose(1, 2).reshape(-1, 384, 16, 16)
    grid = torch.nn.functional.avg_pool2d(grid, 2).flatten(2).transpose(1, 2)
    return torch.cat((hidden[:, :1], grid), dim=1)


def wanted_images(rows, arrays, stage, pilot=False):
    assert stage in ('inner', 'outer')
    indices = rows.index if stage == 'outer' else rows.index[rows.fold != 0]
    if pilot:
        assert stage == 'inner'
        candidates = rows[~rows.fold.isin([0, 1])].copy()
        candidates['order'] = candidates['clip'].map(lambda c: hashlib.sha256(('r15|' + c).encode()).hexdigest())
        indices = candidates.sort_values('order').drop_duplicates('source_id').head(12).index
    needed = set(arrays[stage + '_hashes'][indices].flat); needed.discard('')
    return sorted(needed), list(map(int, indices))


def extract(rows, arrays, images, by_hash, weights, out, stage='inner', pilot=False, budget=600):
    assert torch.cuda.is_available(), 'Colab CUDA required; no CPU fallback'
    assert budget > 60
    out = Path(out); torch.set_num_threads(4)
    selection = json.loads((out / 'selection_protocol.json').read_text())
    assert selection['sources'] == source_hashes()
    protocol = {'selection_protocol_sha256': data.sha(out / 'selection_protocol.json'),
                'sources': source_hashes(), 'weights_sha256': dino_probe.WEIGHT_SHA,
                'preprocess': 'RGB JPEG224; original + horizontal mirror; ImageNet mean/std; CUDA fp16',
                'tokens': selection['tokens']}
    data.immutable_json(out / 'token_protocol.json', protocol)
    wanted, indices = wanted_images(rows, arrays, stage, pilot)
    if not pilot:
        cost = json.loads((out / 'cost_pilot.json').read_text())
        assert cost['gate_passed'] and cost['token_protocol_sha256'] == data.sha(out / 'token_protocol.json')
    completed, pending = [], []
    for digest in wanted:
        target, record = out / 'tokens' / (digest + '.npz'), out / 'records' / (digest + '.json')
        if target.exists() and record.exists():
            info = json.loads(record.read_text())
            assert info['input_sha256'] == digest and data.sha(target) == info['sha256']
            assert info['token_protocol_sha256'] == data.sha(out / 'token_protocol.json')
            completed.append(digest)
        else: pending.append(digest)
    reused = len(completed); started = time.monotonic(); timings = []
    net = dino_probe.load_model(Path(weights)) if pending else None
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    if net is not None:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            net(pixel_values=torch.zeros((2, 3, 224, 224), device='cuda'))
        torch.cuda.synchronize()
    with zipfile.ZipFile(images) as archive:
        for offset in range(0, len(pending), 8):
            if time.monotonic() - started > budget - 60: break
            batch_hashes = pending[offset:offset + 8]; pixels = []
            tick = time.monotonic()
            for digest in batch_hashes:
                payload = archive.read(by_hash[digest])
                assert hashlib.sha256(payload).hexdigest() == digest
                image = np.asarray(Image.open(io.BytesIO(payload)).convert('RGB')).copy()
                assert image.shape == (224, 224, 3)
                pixels.extend([image, image[:, ::-1].copy()])
            batch = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).cuda().float() / 255
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                tokens = spatial_tokens(net(pixel_values=(batch - mean) / std).last_hidden_state)
            assert tokens.is_cuda and torch.isfinite(tokens).all() and tokens.shape[1:] == (65, 384)
            values = tokens.half().cpu().numpy().reshape(-1, 2, 65, 384)
            for digest, value in zip(batch_hashes, values):
                target = out / 'tokens' / (digest + '.npz'); atomic_npz(target, tokens=value)
                data.dump(out / 'records' / (digest + '.json'), {'input_sha256': digest,
                          'token_protocol_sha256': data.sha(out / 'token_protocol.json'), 'sha256': data.sha(target)})
                completed.append(digest)
            torch.cuda.synchronize(); timings.append((time.monotonic() - tick) / len(batch_hashes))
            data.dump(out / 'extraction_status.json', {'stage': stage, 'pilot': pilot, 'completed': len(completed),
                      'requested': len(wanted), 'reused': reused, 'seconds': time.monotonic() - started})
    report = {'status': 'complete' if len(completed) == len(wanted) else 'budget_boundary',
              'stage': stage, 'pilot': pilot, 'completed': len(completed), 'requested': len(wanted), 'reused': reused,
              'seconds': time.monotonic() - started, 'gpu': torch.cuda.get_device_name(0),
              'cuda_tokens_verified_this_run': bool(timings), 'selection_indices': indices,
              'token_protocol_sha256': data.sha(out / 'token_protocol.json')}
    if pilot and timings:
        full_count = len(wanted_images(rows, arrays, 'inner')[0])
        estimate = max(timings) * full_count * 1.5
        report.update(estimated_inner_seconds=estimate, estimated_uncompressed_inner_bytes=full_count * 2 * 65 * 384 * 2,
                      gate_passed=report['status'] == 'complete' and estimate < 600)
        data.dump(out / 'cost_pilot.json', report)
    data.dump(out / 'extraction_status.json', report)
    print(json.dumps(report), flush=True)
    return report
