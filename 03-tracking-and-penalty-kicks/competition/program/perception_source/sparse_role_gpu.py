"""Bounded CUDA extraction for sparse static-role experiments; no dense tracker."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

import rgb_gru as rgb
import rgb_gru_data as data
import sparse_role_pilot as sparse

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'version': 1, 'anchor_offsets_s': [-.35, -.16, 0.],
          'selection': 'per-anchor pose-free static pairs; best score also saved as a mask',
          'pixel_size': 224, 'crop_margin': .2, 'encoder': rgb.ENCODER_CONFIG,
          'role_config': sparse.CONFIG['role_config'],
          'detector_config': {**sparse.baseline.CONFIG, 'person_model': 'YOLOX HumanArt m, 640x640, no pose'},
          'uses_pose': False, 'uses_tracking': False, 'uses_field_lines': False}


def choose_anchors(frames):
    end = frames[-1][1]
    indices = sorted({min(range(len(frames)), key=lambda i: abs(frames[i][1] - end - offset))
                      for offset in CONFIG['anchor_offsets_s']})
    return [frames[i] for i in indices]


def make_views(frames, selected):
    images, masks = [], []
    for (_, _, frame), record in zip(frames, selected):
        pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        views = np.zeros((3, 224, 224, 3), np.uint8)
        mask = np.array([True, False, False])
        views[0] = data.letterbox(pixels)
        for i, role in enumerate(['kicker', 'keeper'], 1):
            crop = data.crop_view(pixels, record['roles'][role])
            if crop is not None:
                views[i] = crop; mask[i] = True
        images.append(views); masks.append(mask)
    mask = np.stack(masks)
    best_mask = mask.copy(); best_mask[:, 1:] = False
    candidates = [i for i, r in enumerate(selected) if r['selection'] is not None]
    if candidates:
        best = max(candidates, key=lambda i: selected[i]['selection']['score'])
        best_mask[best, 1:] = mask[best, 1:]
    return np.stack(images), mask, best_mask


class Models:
    def __init__(self):
        import onnxruntime as ort
        from rtmlib import YOLOX
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        assert torch.cuda.is_available(), 'CUDA required'
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        assert 'CUDAExecutionProvider' in ort.get_available_providers()
        self.people = YOLOX(str(ROOT / 'models/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
                            model_input_size=(640, 640), backend='onnxruntime', device='cuda')
        assert self.people.session.get_providers()[0] == 'CUDAExecutionProvider'
        path = ROOT / 'models/grounding-dino-tiny'
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.grounder = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).eval().cuda()
        self.encoder, self.encoder_manifest = rgb.make_encoder(ROOT / 'models/efficientnet-b0')
        self.encoder.eval().cuda()
        self.runtime = {'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
                        'onnxruntime': ort.__version__, 'person_providers': self.people.session.get_providers(),
                        'grounder_device': str(next(self.grounder.parameters()).device),
                        'encoder_device': str(next(self.encoder.parameters()).device),
                        'encoder': self.encoder_manifest}

    def detect(self, frame):
        torch.cuda.synchronize(); start = time.perf_counter()
        h, w = frame.shape[:2]
        persons = [{'bbox': sparse.baseline.clamp_box(box, w, h)} for box in self.people(frame)]
        torch.cuda.synchronize(); person_seconds = time.perf_counter() - start
        start = time.perf_counter()
        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                text=sparse.baseline.CONFIG['prompt'], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = self.grounder(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.2, text_threshold=.2, target_sizes=[(h, w)])[0]
        objects = []
        for box, score, label in zip(result['boxes'], result['scores'], result['text_labels']):
            kind = 'goal' if 'goal' in label and 'ball' not in label else (
                'ball' if 'ball' in label and 'goal' not in label else 'ambiguous')
            objects.append({'bbox': sparse.baseline.clamp_box(box.tolist(), w, h),
                            'score': float(score), 'label': label, 'kind': kind})
        torch.cuda.synchronize()
        return {'width': w, 'height': h, 'persons': persons, 'objects': objects}, {
            'person_seconds': person_seconds, 'grounding_seconds': time.perf_counter() - start}


def process(clip, video, models, out, protocol_sha):
    record_path = out / 'records' / (clip + '.json')
    inputs = {'video_sha256': data.sha(video), 'protocol_sha256': protocol_sha}
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['inputs'] == inputs, 'Resume inputs differ'
        for name, digest in record['files'].items():
            assert data.sha(out / name) == digest, name
        return record
    start = time.perf_counter()
    frames = choose_anchors(sparse.baseline.read_tail(video))
    timing = {'decode_seconds': time.perf_counter() - start, 'person_seconds': 0., 'grounding_seconds': 0.}
    anchors = []
    for index, timestamp, frame in frames:
        detected, costs = models.detect(frame)
        anchors.append({**detected, 'frame_index': index, 'timestamp': timestamp})
        for name, cost in costs.items():
            timing[name] += cost
    tick = time.perf_counter()
    selected = sparse.select(anchors, 'per_anchor')
    images, mask, best_mask = make_views(frames, selected)
    timing['selection_crop_seconds'] = time.perf_counter() - tick
    ticks = np.array([f[1] for f in frames]); indices = np.array([f[0] for f in frames])
    tick = time.perf_counter()
    with torch.inference_mode():
        tensor = torch.from_numpy(images.transpose(0, 1, 4, 2, 3)).float()[None].cuda()
        feature_tensor = rgb.image_features(models.encoder, tensor, torch.from_numpy(mask)[None].cuda())
        assert feature_tensor.is_cuda and torch.isfinite(feature_tensor).all()
        features = feature_tensor[0].cpu().numpy()
    torch.cuda.synchronize(); timing['encoding_seconds'] = time.perf_counter() - tick
    for folder, key, value in [('crops', 'images', images), ('features', 'features', features)]:
        path = out / folder / (clip + '.npz'); path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **{key: value}, view_mask=mask, best_view_mask=best_mask,
                            timestamps=ticks, frame_indices=indices)
    detection_path = out / 'detections' / (clip + '.json')
    data.dump(detection_path, {'clip': clip, 'anchors': anchors, 'selected': selected})
    sheet = Image.new('RGB', (640 * len(frames), 390), 'white')
    for i, (frame, chosen) in enumerate(zip(frames, selected)):
        sheet.paste(sparse.annotated(frame[2], chosen['roles'], f'{clip} | t={frame[1]:.3f}'), (640 * i, 0))
    (out / 'examples').mkdir(exist_ok=True)
    sheet.save(out / 'examples' / (clip + '.jpg'), quality=90)
    timing['total_seconds'] = time.perf_counter() - start
    paths = [f'{folder}/{clip}.{suffix}' for folder, suffix in [('crops', 'npz'), ('features', 'npz'),
                                                              ('detections', 'json'), ('examples', 'jpg')]]
    record = {'clip': clip, 'inputs': inputs, 'timing': timing,
              'valid_pairs': int(mask[:, 1:].all(1).sum()), 'anchors': len(frames),
              'cuda_inference_verified': True, 'files': {name: data.sha(out / name) for name in paths}}
    data.dump(record_path, record)
    return record


def run(args):
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    rows = pd.read_csv(args.sample)
    if args.scope == 'smoke':
        assert len(rows) == 12 and rows.source_id.nunique() == 12 and (rows.fold != 0).all()
    else:
        assert len(rows) == 1299 and rows.groupby('source_id').fold.nunique().max() == 1
        assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    data.immutable_json(out / 'run_scope.json', {'scope': args.scope, 'sample_sha256': data.sha(args.sample),
                        'sample': rows[['clip', 'source_id', 'fold']].to_dict('records')})
    print(json.dumps({'stage': 'loading_models', 'clips': len(rows)}), flush=True)
    models = Models()
    data.dump(out / 'runtime.json', models.runtime)
    protocol = {'config': CONFIG, 'encoder': models.encoder_manifest,
                'models': json.loads((ROOT / 'selected_asset_manifest.json').read_text()),
                'source_sha256': {Path(p).name: data.sha(p) for p in [__file__, sparse.__file__,
                                  sparse.baseline.__file__, data.__file__, rgb.__file__]}}
    data.immutable_json(out / 'protocol.json', protocol); protocol_sha = data.sha(out / 'protocol.json')
    records = []
    try:
        for clip in rows['clip']:
            if time.monotonic() - started > args.max_seconds - 120:
                data.dump(out / 'status.json', {'stage': 'checkpointed', 'completed': len(records), 'total': len(rows)})
                break
            record = process(clip, args.raw / 'train' / clip, models, out, protocol_sha)
            records.append(record)
            print(json.dumps({'stage': 'sparse_extraction', 'completed': len(records), 'total': len(rows),
                              'clip': clip, 'timing': record['timing'], 'valid_pairs': record['valid_pairs']}), flush=True)
        else:
            data.dump(out / 'status.json', {'stage': 'complete', 'completed': len(rows), 'total': len(rows),
                                           'elapsed_seconds': time.monotonic() - started})
        data.dump(out / 'summary.json', {'records': records, 'note': 'Availability is not role accuracy; no classification training in smoke.'})
    except Exception as exc:
        data.dump(out / 'status.json', {'stage': 'error', 'error': str(exc), 'completed': len(records)})
        raise
    finally:
        data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*')
                  if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scope', choices=['smoke', 'public'], default='smoke')
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--max-seconds', type=int, default=900)
    run(p.parse_args())
