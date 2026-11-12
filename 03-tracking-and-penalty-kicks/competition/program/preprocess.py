"""GPU entry points for the preserved automatic perception implementations.

These are optional recomputations. The submitted model's exact feature cache
is preserved separately; newly measured results never replace it silently.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from run import HERE, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['legacy', 'field', 'tracks'], required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--population', choices=['public', 'private', 'all'], default='private')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    with gzip.open(HERE / 'assets/automatic_perception.json.gz', 'rt') as f:
        rows = json.load(f)
    rows = [r for r in rows if args.population == 'all' or
            ('public' if r['clip'].endswith('_public.mp4') else 'private') == args.population]
    if args.limit:
        rows = rows[:args.limit]
    videos = {r['clip']: str(args.dataset / ('train' if r['clip'].endswith('_public.mp4') else 'val') / r['clip']) for r in rows}
    for r in rows:
        assert sha(videos[r['clip']]) == r['anchors'][0]['video_sha256']
        assert r['clip'] != '0526_public.mp4'
        if r['clip'] == '0058_public.mp4':
            assert max(a['frame_index'] for a in r['anchors']) <= 68
    if args.inspect:
        print(json.dumps(dict(stage=args.stage, videos_verified=len(rows), CUDA_required=True,
                              reference_cache_overwritten=False), indent=2)); return
    import torch
    assert torch.cuda.is_available(), 'CUDA required for perception; no CPU bulk fallback'
    sys.path.insert(0, str(HERE / 'perception_source'))
    import rgb_gru_data as data
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'tracks':
        import stable_role_motion as tracker
        inputs = args.out / 'inputs'; inputs.mkdir(exist_ok=True)
        data.dump(inputs / 'samples.json', rows)
        data.dump(inputs / 'input_manifest.json', {'files': {'samples.json': sha(inputs / 'samples.json')}})
        tracker.run(inputs, args.out / 'tracks', args.pretrained, videos, budget=args.seconds)
    elif args.stage == 'field':
        import cv2
        from calibrated_private_inference import FieldCalibrator
        field = json.loads((HERE / 'assets/field_contract.json').read_text())
        roots = [p for p in (args.pretrained / 'pnlcalib_source').iterdir() if (p / 'config').is_dir()]
        assert len(roots) == 1
        protocol = sha(HERE / 'assets/field_contract.json')
        model = FieldCalibrator(roots[0], args.pretrained / 'field/weights', field['weights'], args.out, protocol)
        start = time.monotonic()
        for row in rows:
            if time.monotonic() - start > args.seconds - 60:
                break
            sample = row['anchors'][-1]; path = args.out / 'records' / (row['clip'] + '.json')
            if path.exists():
                continue
            cap = cv2.VideoCapture(videos[row['clip']])
            try:
                for _ in range(sample['frame_index'] + 1):
                    ok, frame = cap.read(); assert ok
            finally:
                cap.release()
            model.infer(sample, frame)
    else:
        import sparse_role_gpu as sparse
        import private_penalty_features as legacy
        model = legacy.Models(args.pretrained, args.pretrained / 'nlf.torchscript', HERE / 'assets/models', args.out)
        original = sparse.choose_anchors; start = time.monotonic()
        import numpy as np
        public = np.load(HERE / 'assets/features/public_trees.npz')
        private = np.load(HERE / 'assets/features/private_trees.npz')
        feet = {str(c): ['left', 'right', 'unknown'][int(x[186:189].argmax())]
                for a in [public, private] for c, x in zip(a['clips'], a['fold0'])}
        for sample in rows:
            if time.monotonic() - start > args.seconds - 120:
                break
            clip = sample['clip']
            sparse.choose_anchors = (lambda frames: original([f for f in frames if f[0] <= 68])) if clip == '0058_public.mp4' else original
            foot = feet[clip]
            legacy.process(dict(clip=clip, preferred_foot=foot), videos[clip], model, args.out,
                           sha(HERE / 'perception_source/source_manifest.json'))
        sparse.choose_anchors = original
    print('Measured outputs saved separately. Compare their provenance before replacing any frozen feature inputs.')


if __name__ == '__main__':
    main()
