"""Regenerate native crops and neural features from externally supplied videos.

The saved observations are automatic detection/tracking/calibration results,
not manual boxes. Reusing them is part of the reproducibility contract.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from run import HERE, SEEDS, sha
from sam_core import dump


def load_rows(population):
    rows = json.loads((HERE / 'assets/observations.json').read_text())
    return [r for r in rows if population == 'all' or r['population'] == population]


def video_path(dataset, row):
    folder = 'train' if row['population'] == 'public' else 'val'
    path = dataset / folder / row['clip']
    if not path.is_file():
        raise FileNotFoundError(f'Expected external competition video: {path}')
    if sha(path) != row['video_sha256']:
        raise ValueError(f'Video differs from the frozen experiment: {row["clip"]}')
    return path


def native_crop(frame, transform):
    """Use the exact wide480 -> central384 procedure of the training crops."""
    import cv2
    a = np.asarray(transform, np.float64)
    scale = a[0, 0]; side = 384 / scale
    origin = -a[:2, 2] / scale
    wide_origin = origin - side * .125
    matrix = np.array([[scale, 0, -wide_origin[0]*scale],
                       [0, scale, -wide_origin[1]*scale]], np.float32)
    return cv2.warpAffine(frame, matrix, (480, 480), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)[48:432, 48:432]


def crops(dataset, work, rows, strict=False):
    import cv2
    cv2.setNumThreads(1)
    reports = []
    for row in rows:
        path = video_path(dataset, row)
        if row['clip'] == '0058_public.mp4':
            assert max(f['frame_index'] for f in row['frames']) <= 68
        todo = {}
        for slot, f in enumerate(row['frames']):
            if f['available']:
                todo.setdefault(f['frame_index'], []).append((slot, f))
        cap = cv2.VideoCapture(str(path)); time_origins = []
        try:
            for index in range(max(todo, default=-1) + 1):
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {row["clip"]} frame {index}')
                if index not in todo:
                    continue
                for slot, f in todo[index]:
                    time_origins.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000 - row['times'][slot])
                    output = work / f['file']
                    output.parent.mkdir(parents=True, exist_ok=True)
                    pixels = native_crop(image, f['original_to_crop'])
                    assert pixels.shape == (384, 384, 3)
                    assert cv2.imwrite(str(output), pixels)
                    exact = sha(output) == f['image_sha256']
                    reports.append(dict(clip=row['clip'], slot=slot, exact_png=exact,
                                        pixel_sha256=__import__('hashlib').sha256(pixels.tobytes()).hexdigest()))
                    if strict and not exact:
                        raise ValueError('PNG differs from the original OpenCV/FFmpeg environment; '
                                         'use the reference numeric cache or match the recorded runtime')
        finally:
            cap.release()
        if time_origins:
            assert max(time_origins) - min(time_origins) < .003, 'Observed relative frame times changed'
    dump(work / 'crop_report.json', dict(opencv=cv2.__version__, frames=len(reports),
                                        exact_png=sum(r['exact_png'] for r in reports), records=reports))


def rgb(work, pretrained, models, rows, features):
    import torch
    from PIL import Image
    import body_visual_probe as visual
    import human_rgb_adaptation as adapt
    assert torch.cuda.is_available(), 'CUDA required for the full pretrained image encoder'
    contract = json.loads((HERE / 'assets/model_contract.json').read_text())['base_pretrained']
    assert sha(pretrained / 'model.ckpt') == contract['weights_sha256']
    model, audit = visual.encoder(pretrained / 'model.ckpt', 'sam_body_dinov3', contract['dinov3_revision'])
    model.half().eval(); adapt.inject(model, 8, 8, 8)
    metadata = {r['clip']: r for r in json.loads((HERE / 'assets/rgb_samples.json').read_text())}
    private_rows = [r for r in rows if r['population'] == 'private']
    for seed in SEEDS:
        state = torch.load(models / f'rgb/{seed}/checkpoint.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            for name, p in adapt.adapter_parameters(model).items():
                p.copy_(state['adapters'][name].to(p))
        values = []
        for row in private_rows:
            if not metadata[row['clip']]['available']:
                values.append(np.zeros(5120, np.float32)); continue
            image = np.asarray(Image.open(work / row['frames'][-1]['file']).convert('RGB')).copy()
            if row['flip']:
                image = image[:, ::-1].copy()
            x = torch.from_numpy(image).permute(2, 0, 1).float() / 255
            x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            with torch.inference_mode():
                value = adapt.full_features(model, x[None].cuda().half())
            assert value.is_cuda
            values.append(value.float().cpu().numpy()[0])
        features.mkdir(parents=True, exist_ok=True)
        np.save(features / f'rgb_{seed}.npy', np.array(values, np.float32))
    dump(work / 'rgb_encoder_audit.json', dict(actual_cuda=True, rows=len(private_rows), loaded=audit))


def sam(work, pretrained, rows, features):
    import os
    import sys
    import torch
    import native_body as native
    assert torch.cuda.is_available(), 'CUDA required for the native SAM body model'
    contract = json.loads((HERE / 'assets/sam_contract.json').read_text())['feature_contract']
    official = contract['official_model']
    os.environ['MOMENTUM_ENABLED'] = '0'
    os.environ['WANDB_MODE'] = 'disabled'
    source_path = pretrained / 'sam_source_path.txt'
    if source_path.exists():
        sys.path.insert(0, source_path.read_text().strip())
    for key, name in [('checkpoint', 'model.ckpt'), ('mhr', 'mhr_model.pt')]:
        assert sha(pretrained / name) == official[key]['sha256']
    model, estimator = native.load_native(pretrained, work)
    for population in ['public', 'private']:
        selected = [r for r in rows if r['population'] == population]
        if not selected:
            continue
        values, masks = [], []
        for original in selected:
            row = dict(original)
            tokens = np.zeros((5, 1024), np.float32); mask = np.zeros(5, bool)
            if row['eligible_input']:
                for slot, f in enumerate(row['frames']):
                    if not f['available']:
                        continue
                    frame = dict(f)
                    # The raw decode audit records environment differences.
                    # Native extraction binds its actual input, never a stale hash.
                    frame['image_sha256'] = sha(work / f['file'])
                    tokens[slot], _ = native.infer(model, estimator, work, row, frame)
                    mask[slot] = True
            values.append(tokens); masks.append(mask)
        features.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(features / f'sam_{population}.npz', clips=np.array([r['clip'] for r in selected]),
                            tokens=np.array(values), mask=np.array(masks),
                            times=np.array([r['times'] for r in selected], np.float32),
                            axis=np.array([r['axis'] for r in selected], np.float32))
    dump(work / 'sam_encoder_audit.json', dict(actual_cuda=True, clips=len(rows), frozen_encoder=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Competition root containing train/ and val/')
    parser.add_argument('--work', type=Path, required=True, help='External writable directory for regenerated crops')
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--models', type=Path, default=HERE / 'assets/models')
    parser.add_argument('--features-out', type=Path, help='Separate regenerated feature directory; never overwrite the reference cache')
    parser.add_argument('--population', choices=['public', 'private', 'all'], default='all')
    parser.add_argument('--stage', choices=['crops', 'rgb', 'sam', 'all'], default='crops')
    parser.add_argument('--limit', type=int, help='Smoke test only; partial features cannot be used as the 900-row submission input')
    parser.add_argument('--strict-images', action='store_true')
    args = parser.parse_args(); rows = load_rows(args.population)
    if args.limit:
        rows = rows[:args.limit]
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage in ['crops', 'all']:
        crops(args.dataset, args.work, rows, args.strict_images)
    if args.stage != 'crops':
        if args.pretrained is None or args.features_out is None:
            parser.error('Neural extraction needs --pretrained and --features-out')
        if args.features_out.resolve() == (HERE / 'assets/features').resolve():
            parser.error('Use a separate --features-out directory to protect the reference cache')
    if args.stage in ['rgb', 'all']:
        rgb(args.work, args.pretrained, args.models, rows, args.features_out)
    if args.stage in ['sam', 'all']:
        sam(args.work, args.pretrained, rows, args.features_out)


if __name__ == '__main__':
    main()
