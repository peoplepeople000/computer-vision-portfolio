"""R7: frozen DINOv2 global and pose-guided lower-body appearance features."""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import rgb_gru_data as data

MODEL = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
WEIGHT_SHA = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'


def lower_body_pool(tokens, points, scores):
    """Confidence-weighted patch pooling, using only valid hips/legs/feet."""
    assert tokens.shape[1] == 256 and points.shape[1:] == (23, 2)
    points, scores = points[:, 11:23], scores[:, 11:23]
    valid = (scores > .3) & torch.isfinite(scores) & torch.isfinite(points).all(-1) & ((points >= 0) & (points < 224)).all(-1)
    cell = (torch.nan_to_num(points) / 14).floor().long().clamp(0, 15)
    index = cell[..., 1] * 16 + cell[..., 0]
    selected = tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))
    weight = torch.where(valid, scores.clamp(0, 1), 0).float()
    pooled = (selected.float() * weight[..., None]).sum(1) / weight.sum(1).clamp_min(1e-6)[:, None]
    return pooled, valid.any(1), valid.float().mean(1)


def load_model(directory):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
        if not (directory / name).exists():
            hf_hub_download(MODEL, name, revision=REVISION, local_dir=directory, token=False)
    assert data.sha(directory / 'model.safetensors') == WEIGHT_SHA
    processor = json.loads((directory / 'preprocessor_config.json').read_text())
    assert processor['image_mean'] == [.485, .456, .406] and processor['image_std'] == [.229, .224, .225]
    net = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False).eval().cuda()
    net.requires_grad_(False)
    assert net.config.hidden_size == 384 and net.config.patch_size == 14
    return net


def extract(bundle, pose_dir, weights, out, pilot=False, budget=900):
    import pandas as pd
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4)
    rows = pd.read_csv(bundle / 'public_folds.csv')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items(): assert data.sha(bundle / name) == digest
    protocol = {'source_sha256': data.sha(__file__), 'model': MODEL, 'revision': REVISION,
                'weights_sha256': WEIGHT_SHA, 'bundle_sha256': data.sha(bundle / 'manifest.json'),
                'pose_protocol_sha256': data.sha(pose_dir / 'protocol.json'),
                'preprocessing': 'Existing JPEG RGB 224x224, no resize/center crop, published ImageNet mean/std.',
                'precision': 'CUDA fp16 autocast; exported features float32',
                'selection': 'Latest masked actor crop with any pose confidence >0.3; no identity tracking claim.',
                'pooling': 'Global CLS vs confidence-weighted lower-body patch tokens, joint indices 11..22.',
                'gpu': torch.cuda.get_device_name(0)}
    data.immutable_json(out / 'protocol.json', protocol)
    records = manifest['records']
    assert list(rows['clip']) == [r['clip'] for r in records] and len(rows) == 1299
    if pilot:
        candidates = rows[~rows.fold.isin([0, 1])].copy()
        candidates['order'] = candidates['clip'].map(lambda c: hashlib.sha256(c.encode()).hexdigest())
        indices = candidates.sort_values('order').drop_duplicates('source_id').head(12).index.tolist()
    else: indices = list(range(len(rows)))
    net = load_model(weights)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        net(pixel_values=torch.zeros((2, 3, 224, 224), device='cuda'))
    torch.cuda.synchronize(); started = time.monotonic()
    inference_times, images_done, completed, reused = [], 0, 0, 0
    with zipfile.ZipFile(bundle / 'images.zip') as z:
        for offset in range(0, len(indices), 16):
            if time.monotonic() - started > budget - 60: break
            batch_records, images, points, scores, destinations = [], [], [], [], []
            for index in indices[offset:offset+16]:
                original = records[index]; clip = original['clip']
                pose_path = pose_dir / 'features' / (clip + '.npz')
                pose_record = json.loads((pose_dir / 'records' / (clip + '.json')).read_text())
                assert data.sha(pose_path) == pose_record['sha256'] and pose_record['input'] == original
                inputs = {'original': original, 'pose_sha256': pose_record['sha256']}
                target, record_file = out / 'features' / (clip + '.npz'), out / 'records' / (clip + '.json')
                if target.exists() and record_file.exists():
                    saved = json.loads(record_file.read_text())
                    assert saved['inputs'] == inputs and data.sha(target) == saved['sha256']
                    completed += 1; reused += 1; continue
                item = {'global': np.zeros((2, 384), np.float32), 'lower_body': np.zeros((2, 384), np.float32),
                        'mask': np.zeros((2, 2), bool), 'age': np.zeros(2, np.float32),
                        'joint_fraction': np.zeros(2, np.float32), 'anchor': np.full(2, -1, np.int64)}
                local_index = len(batch_records)
                with np.load(pose_path) as pose:
                    for role in range(2):
                        valid = np.flatnonzero(pose['mask'][:, role] & (pose['scores'][:, role, :23] > .3).any(1))
                        if not len(valid): continue
                        t = int(valid[-1]); key = f'{clip}/{t}_{role}.jpg'; payload = z.read(key)
                        assert hashlib.sha256(payload).hexdigest() == manifest['image_hashes'][key]
                        image = np.asarray(Image.open(io.BytesIO(payload)).convert('RGB')).copy()
                        assert image.shape == (224, 224, 3)
                        images.append(torch.from_numpy(image).permute(2, 0, 1))
                        points.append(pose['points'][t, role, :23].copy()); scores.append(pose['scores'][t, role, :23].copy())
                        destinations.append((local_index, role))
                        item['anchor'][role] = t; item['age'][role] = pose['timestamps'][-1] - pose['timestamps'][t]
                batch_records.append((item, inputs, target, record_file))
            if images:
                pixels = torch.stack(images).cuda().float() / 255.
                torch.cuda.synchronize(); tick = time.monotonic()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    tokens = net(pixel_values=(pixels - mean) / std).last_hidden_state
                assert tokens.is_cuda and tokens.shape == (len(images), 257, 384) and torch.isfinite(tokens).all()
                pooled, pool_mask, fraction = lower_body_pool(tokens[:, 1:], torch.tensor(np.stack(points), device='cuda'), torch.tensor(np.stack(scores), device='cuda'))
                torch.cuda.synchronize(); inference_times.append(time.monotonic() - tick)
                global_values, local_values = tokens[:, 0].float().cpu().numpy(), pooled.cpu().numpy()
                for j, (local_index, role) in enumerate(destinations):
                    item = batch_records[local_index][0]
                    item['global'][role] = global_values[j]; item['lower_body'][role] = local_values[j]
                    item['mask'][role] = [True, bool(pool_mask[j])]; item['joint_fraction'][role] = float(fraction[j])
                images_done += len(images)
            for item, inputs, target, record_file in batch_records:
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **item)
                data.dump(record_file, {'inputs': inputs, 'sha256': data.sha(target)})
                completed += 1
            data.dump(out / ('pilot_status.json' if pilot else 'status.json'),
                      {'stage': 'running', 'completed': completed, 'requested': len(indices), 'reused': reused,
                       'images_encoded': images_done, 'seconds': time.monotonic() - started})
    report = {'stage': 'complete' if completed == len(indices) else 'budget_boundary', 'pilot': pilot,
              'completed': completed, 'requested': len(indices), 'reused': reused, 'images_encoded': images_done,
              'seconds': time.monotonic() - started, 'gpu': torch.cuda.get_device_name(0),
              'cuda_tokens_verified': bool(images_done), 'inference_seconds': float(sum(inference_times)),
              'seconds_per_image': float(sum(inference_times) / max(1, images_done)),
              'max_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
              'pilot_sources': rows.iloc[indices].source_id.tolist() if pilot else None,
              'estimated_full_seconds': (time.monotonic() - started) * 1299 / max(1, completed - reused) * 1.5 if pilot and completed > reused else None}
    data.dump(out / ('pilot_status.json' if pilot else 'status.json'), report)
    print(json.dumps(report), flush=True)
    return report
