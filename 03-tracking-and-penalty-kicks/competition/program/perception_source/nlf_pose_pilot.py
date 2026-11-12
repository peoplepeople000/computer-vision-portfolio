"""R18 frozen official NLF-L: bounded human-pose pilot on existing crops."""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

MODEL_URL = 'https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.patch6.torchscript'
MODEL_SHA = '2f9e377817acdd47e35224aeb70299b3f53ab574e0829372a094920515352a65'
MODEL_SIZE = 549527206
JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
               'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
               'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
         (6, 9), (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (12, 15),
         (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def observed_box(points, scores):
    """A box from cached visible RTMW joints, with fixed 15% context margin."""
    points, scores = np.asarray(points), np.asarray(scores)
    valid = (scores > .3) & np.isfinite(scores) & np.isfinite(points).all(1)
    valid &= ((points >= 0) & (points <= 224)).all(1)
    if valid.sum() < 6: return None
    low, high = points[valid].min(0), points[valid].max(0)
    margin = np.maximum((high - low) * .15, 4.)
    low, high = np.maximum(low - margin, 0), np.minimum(high + margin, 224)
    if np.min(high - low) < 12: return None
    return np.r_[low, high - low].astype(np.float32)


def projection_error(projected, points, scores, height):
    """Agreement with cached RTMW, NOT accuracy against ground truth."""
    nlf = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])
    rtmw = np.array([11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10])
    a, b = np.asarray(projected)[nlf], np.asarray(points)[rtmw]
    valid = (np.asarray(scores)[rtmw] > .3) & np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if valid.sum() < 6: return None
    return float(np.median(np.linalg.norm(a[valid] - b[valid], axis=1)) / max(float(height), 1.))


def run(pilot_json, image_archive, out, budget=600):
    import requests
    import torch
    import torchvision  # Register operators needed by the official TorchScript.
    from PIL import Image, ImageDraw
    assert torch.cuda.is_available(), 'Kaggle CUDA required'
    torch.set_num_threads(4); torch.manual_seed(6643)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(pilot_json).read_text()); assert spec['source_folds'] == [2, 3, 4]
    assert len(spec['samples']) == 12 and len({s['source_id'] for s in spec['samples']}) == 12
    assert sha(image_archive) == spec['image_archive_sha256']
    weights = out / 'nlf_l_multi_0.3.2.patch6.torchscript'
    if not weights.exists():
        temporary = weights.with_suffix('.download')
        with requests.get(MODEL_URL, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024): stream.write(chunk)
        assert temporary.stat().st_size == MODEL_SIZE and sha(temporary) == MODEL_SHA
        temporary.replace(weights)
    assert sha(weights) == MODEL_SHA
    protocol = {'source_sha256': sha(__file__), 'pilot_sha256': sha(pilot_json),
                'model_url': MODEL_URL, 'model_sha256': MODEL_SHA,
                'default_crop_fov_degrees': 55., 'num_aug': 1, 'internal_batch_size': 4,
                'box': 'Cached RTMW visible-joint extent + fixed 15% margin; no detector call.',
                'scope': '12 training sources, latest existing kicker crop. No classifier or held-out labels.',
                'limitation': 'Uncalibrated crop FOV; root-relative human geometry only. No field reconstruction.'}
    dump(out / 'protocol.json', protocol); dump(out / 'pilot_input.json', spec)
    started = time.monotonic(); net = torch.jit.load(str(weights), map_location='cuda').eval()
    schemas = {name: str(net._c._get_method(name).schema) for name in net._c._method_names()}
    dump(out / 'model_methods.json', schemas)
    assert 'estimate_poses_batched' in schemas and 'get_weights_for_canonical_points' in schemas
    canonical = net.cano_all['smpl']
    assert canonical.shape == (1048, 3), tuple(canonical.shape)
    joints = canonical[-24:].cuda().float()
    with torch.inference_mode(), torch.device('cuda'):
        field_weights = net.get_weights_for_canonical_points(joints)
    dump(out / 'skeleton.json', {'names': JOINT_NAMES, 'edges': EDGES,
                                'canonical_points': joints.cpu().tolist(), 'canonical_shape': list(canonical.shape)})
    tiles, records, times = [], [], []
    with zipfile.ZipFile(image_archive) as archive:
        for index, sample in enumerate(spec['samples']):
            if time.monotonic() - started > budget - 60: raise TimeoutError('Pilot saving reserve; completed records preserved')
            payload = archive.read(sample['key']); assert hashlib.sha256(payload).hexdigest() == sample['image_sha256']
            image = Image.open(io.BytesIO(payload)).convert('RGB'); pixels = np.asarray(image).copy()
            assert pixels.shape == (224, 224, 3)
            p, s = np.asarray(sample['points']), np.asarray(sample['scores'])
            box = observed_box(p, s)
            record = {'clip': sample['clip'], 'image_sha256': sample['image_sha256'], 'usable_box': box is not None}
            if box is None:
                records.append(record); continue
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda()
            boxes = [torch.from_numpy(box)[None].cuda()]
            torch.cuda.synchronize(); tick = time.monotonic()
            with torch.inference_mode(), torch.device('cuda'):
                result = net.estimate_poses_batched(tensor, boxes, field_weights,
                    default_fov_degrees=55., num_aug=1, internal_batch_size=4)
            torch.cuda.synchronize(); seconds = time.monotonic() - tick; times.append(seconds)
            assert len(result['poses3d']) == 1 and result['poses3d'][0].shape == (1, 24, 3)
            a, b, u = result['poses3d'][0][0], result['poses2d'][0][0], result['uncertainties'][0][0]
            assert a.is_cuda and torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(u).all()
            a, b, u = a.cpu().numpy(), b.cpu().numpy(), u.cpu().numpy()
            torso = float(np.linalg.norm((a[16] + a[17]) * .5 - (a[1] + a[2]) * .5))
            error = projection_error(b, p, s, box[3])
            record.update(seconds=seconds, cuda_verified=True, projected_joint_median_error_fraction=error,
                          torso_mm=torso, median_uncertainty_mm=float(np.median(u) * 1000), box=box.tolist())
            np.savez_compressed(out / (sample['clip'] + '.npz'), points3d=a, points2d=b, uncertainty=u,
                                box=box, image_sha256=sample['image_sha256'])
            record['output_sha256'] = sha(out / (sample['clip'] + '.npz'))
            records.append(record); dump(out / 'records.json', records)
            tile = image.copy(); draw = ImageDraw.Draw(tile)
            for first, second in EDGES: draw.line([tuple(b[first]), tuple(b[second])], fill=(0, 255, 255), width=2)
            for point in p[s > .3]: draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(255, 200, 0))
            label = Image.new('RGB', (224, 250), 'white'); label.paste(tile, (0, 0)); draw = ImageDraw.Draw(label)
            draw.text((4, 227), f'{sample["clip"]} error={error:.3f}' if error is not None else sample['clip'], fill='black')
            tiles.append(label)
            print(json.dumps({'stage': 'pilot_clip', 'completed': index+1, **record}), flush=True)
    canvas = Image.new('RGB', (224 * 4, 250 * 3), 'white')
    for i, tile in enumerate(tiles): canvas.paste(tile, ((i % 4)*224, (i // 4)*250))
    canvas.save(out / 'pilot_gallery.jpg', quality=95)
    valid = [r for r in records if r.get('cuda_verified')]
    errors = [r['projected_joint_median_error_fraction'] for r in valid if r['projected_joint_median_error_fraction'] is not None]
    report = {'status': 'pilot_complete', 'clips': len(records), 'cuda_predictions': len(valid),
              'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
              'seconds': time.monotonic()-started, 'median_forward_seconds': float(np.median(times)),
              'max_cuda_memory_bytes': int(torch.cuda.max_memory_allocated()),
              'estimated_1299_one_view_seconds': float(max(times[2:] or times) * 1299 * 1.5),
              'warmup_observations_excluded': min(2, len(times)),
              'raw_uncertainty_units': 'metres in estimate_poses API; summary converts to millimetres',
              'median_projection_agreement_error_fraction': float(np.median(errors)) if errors else None,
              'quality_gate': 'Inspect gallery and uncertainty before expanding; agreement is not ground truth accuracy.',
              'classifier_trained': False, 'new_score': None, 'preprocessing_rerun': False}
    dump(out / 'summary.json', report); print(json.dumps(report), flush=True)
    return report
