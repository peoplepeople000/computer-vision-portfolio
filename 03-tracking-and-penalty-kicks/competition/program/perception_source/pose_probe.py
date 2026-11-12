"""R3 sparse pose extraction on existing crops; no person detection or tracking."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import rgb_gru_data as data

WEIGHT_SHA = '9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58'
WEIGHT_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip'


def pack(cache, folds, out):
    """JPEG transfer cache: identical frozen crop selection, explicitly lossy pixels."""
    started = time.monotonic(); rows = pd.read_csv(folds)
    assert len(rows) == rows['clip'].nunique() == 1299
    assert rows.groupby('source_id').fold.nunique().max() == 1
    assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    origin = json.loads((cache / 'artifact_manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(out / 'public_folds.csv', index=False)
    pilot = rows[~rows.fold.isin([0, 1])].copy()
    pilot['rank'] = pilot['clip'].map(lambda c: hashlib.sha256(('pose-r3|' + c).encode()).hexdigest())
    pilot = pilot.sort_values('rank').drop_duplicates('source_id').head(12)['clip'].tolist()
    records, image_hashes = [], {}
    with zipfile.ZipFile(out / 'images.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
        for clip in rows['clip']:
            name = f'crops/{clip}.npz'; digest = data.sha(cache / name)
            assert digest == origin[name]
            with np.load(cache / name, allow_pickle=False) as a:
                mask = a['view_mask'][:, 1:]; images = a['images'][:, 1:]
                assert images.shape == (3, 2, 224, 224, 3)
                for t, role in np.argwhere(mask):
                    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(images[t, role], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    assert ok
                    key = f'{clip}/{t}_{role}.jpg'; payload = encoded.tobytes()
                    archive.writestr(key, payload)
                    image_hashes[key] = hashlib.sha256(payload).hexdigest()
                records.append({'clip': clip, 'mask': mask.tolist(), 'timestamps': a['timestamps'].tolist(),
                                'crop_sha256': digest})
    manifest = {'version': 1, 'records': records, 'image_hashes': image_hashes,
                'pilot_clips': pilot, 'source_sha256': data.sha(__file__),
                'weight_sha256': WEIGHT_SHA, 'weight_url': WEIGHT_URL,
                'pixel_encoding': 'JPEG quality 95 from frozen 224px crops; lossy transfer',
                'files': {n: data.sha(out / n) for n in ['images.zip', 'public_folds.csv']}}
    data.dump(out / 'manifest.json', manifest)
    return {'clips': len(records), 'images': len(image_hashes), 'bytes': (out / 'images.zip').stat().st_size,
            'packing_seconds': time.monotonic() - started, 'pilot_clips': pilot}


def extract(bundle, weights, out, full=False, budget=1200):
    import torch
    import onnxruntime as ort
    from rtmlib import RTMPose
    assert torch.cuda.is_available(), 'CUDA required'
    if hasattr(ort, 'preload_dlls'): ort.preload_dlls()
    assert 'CUDAExecutionProvider' in ort.get_available_providers()
    cv2.setNumThreads(1); torch.set_num_threads(4)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items(): assert data.sha(bundle / name) == digest
    assert manifest['source_sha256'] == data.sha(__file__)
    assert data.sha(weights) == WEIGHT_SHA
    protocol = {'source_sha256': data.sha(__file__), 'bundle_sha256': data.sha(bundle / 'manifest.json'),
                'weight_sha256': WEIGHT_SHA, 'joints': list(range(23)),
                'coordinate_system': '224px role crop, not field coordinates',
                'uses_tracking': False, 'uses_detector': False}
    data.immutable_json(out / 'protocol.json', protocol)
    pose = RTMPose(str(weights), model_input_size=(192, 256), backend='onnxruntime', device='cuda')
    options = ort.SessionOptions(); options.enable_profiling = True
    options.profile_file_prefix = str(out / 'cuda_profile')
    pose.session = ort.InferenceSession(str(weights), sess_options=options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    assert pose.session.get_providers()[0] == 'CUDAExecutionProvider'
    started = time.monotonic(); inference_times = []; finished = []; profiled = False
    target = manifest['records'] if full else [r for r in manifest['records'] if r['clip'] in manifest['pilot_clips']]
    with zipfile.ZipFile(bundle / 'images.zip') as archive:
        for record in target:
            clip = record['clip']; saved = out / 'records' / (clip + '.json'); feature = out / 'features' / (clip + '.npz')
            if saved.exists():
                previous = json.loads(saved.read_text())
                assert previous['input'] == record and previous['sha256'] == data.sha(feature)
                finished.append(previous); continue
            if time.monotonic() - started > budget - 60: break
            tick = time.monotonic(); points = np.zeros((3, 2, 23, 2), np.float32)
            scores = np.zeros((3, 2, 23), np.float32); mask = np.array(record['mask'], bool)
            times = []
            for t, role in np.argwhere(mask):
                key = f'{clip}/{t}_{role}.jpg'; payload = archive.read(key)
                assert hashlib.sha256(payload).hexdigest() == manifest['image_hashes'][key]
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                begin = time.perf_counter(); keypoints, confidence = pose(image)
                elapsed = time.perf_counter() - begin
                assert keypoints.shape == (1, 133, 2) and confidence.shape == (1, 133)
                assert np.isfinite(keypoints).all() and np.isfinite(confidence).all()
                points[t, role] = keypoints[0, :23]; scores[t, role] = confidence[0, :23]
                if not profiled:
                    profile = json.loads(Path(pose.session.end_profiling()).read_text())
                    gpu_events = [e for e in profile if e.get('args', {}).get('provider') == 'CUDAExecutionProvider']
                    assert gpu_events, 'No actual CUDA node execution recorded'
                    data.dump(out / 'runtime.json', {'gpu': torch.cuda.get_device_name(0), 'providers': pose.session.get_providers(),
                              'cuda_node_events': len(gpu_events), 'onnxruntime': ort.__version__, 'rtm_shape': list(keypoints.shape)})
                    profiled = True
                else:
                    inference_times.append(elapsed); times.append(elapsed)
            feature.parent.mkdir(parents=True, exist_ok=True)
            with feature.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, points=points, scores=scores, mask=mask, timestamps=np.array(record['timestamps']))
            feature.with_suffix('.tmp').replace(feature)
            result = {'input': record, 'sha256': data.sha(feature), 'inference_seconds': times,
                      'clip_seconds': time.monotonic() - tick,
                      'body_visible_fraction': float((scores[:, :, :17][mask] > .3).mean()) if mask.any() else 0.,
                      'foot_visible_fraction': float((scores[:, :, 17:][mask] > .3).mean()) if mask.any() else 0.}
            data.dump(saved, result); finished.append(result)
            print(json.dumps({'stage': 'pose', 'completed': len(finished), 'target': len(target), 'clip': clip,
                              'seconds': result['clip_seconds']}), flush=True)
    total_images = len(manifest['image_hashes'])
    report = {'stage': 'complete' if len(finished) == len(target) else 'checkpointed', 'scope': 'full' if full else 'pilot',
              'completed': len(finished), 'target': len(target), 'seconds': time.monotonic() - started,
              'median_inference_seconds': float(np.median(inference_times)) if inference_times else None,
              'estimated_all_inference_seconds': float(np.median(inference_times) * total_images) if inference_times else None,
              'total_images': total_images, 'records': finished}
    data.dump(out / ('full_status.json' if full else 'pilot_status.json'), report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'extract'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--folds', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.action == 'pack': print(json.dumps(pack(args.cache, args.folds, args.out), indent=2))
    else: extract(args.cache, args.weights, args.out, full=args.full)
