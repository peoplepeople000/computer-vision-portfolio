"""Pinned, public Cellpose 4 inference with no competition-data fine-tuning.

Use PYTHONPATH=vendor/cellpose4:src with the existing vision interpreter.
Inference never reads ground-truth masks or verified correspondences.
"""
import argparse
import copy
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .data import read_rows, write_rows, read_image, image_path, split_id
from .rle import labels_to_instances
from .cellpose_baseline import confidence_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cpdino'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--modality', choices=['invivo', 'exvivo', 'both'], default='both')
    parser.add_argument('--limit-per-subject', type=int, default=0)
    parser.add_argument('--subject')
    parser.add_argument('--reference', type=Path, default=Path('outputs/stage13_native_relative/preserved/predicted_all.csv'))
    parser.add_argument('--split', choices=['training', 'hidden_test'], default='training')
    args = parser.parse_args()
    from cellpose import models
    if version('cellpose') != '4.2.1':
        raise RuntimeError('Use the isolated, pinned Cellpose 4.2.1 directory')
    torch.set_num_threads(2); torch.manual_seed(2068)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    checkpoint = Path('models/cellpose4')/args.model
    provenance = [json.loads(Path('models/cellpose4/cpdino_provenance.json').read_text())]
    expected = next(r for r in provenance if r['model'] == args.model)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected['sha256']:
        raise ValueError('Public checkpoint changed')
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'checkpoint_sha256': digest, 'cellpose_version': version('cellpose'),
              'dinov3_version': version('dinov3'), 'torch_version': torch.__version__,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'batch_size': 1, 'use_bfloat16': False, 'diameter': None, 'flow_threshold': .4,
              'cellprob_threshold': 0., 'min_size': 15, 'normalization': 'Cellpose default per-channel percentiles 1 and 99',
              'training': 'Official public checkpoint, no competition-data fitting',
              'model_architecture': 'CellposeDINO DINOv3 ViT-L',
              'confidence_maps': 'Cell probability and mask-derived EDT proxy, not the old learned map family'}
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    path = out/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Changed Cellpose 4 configuration')
    path.write_text(json.dumps(config, indent=2)+'\n')
    reference = read_rows(args.reference)
    selected = []
    for subject in sorted({split_id(s)[0] for s in reference}):
        if args.subject and subject != args.subject:
            continue
        ids = sorted(s for s in reference if split_id(s)[0] == subject)
        selected.extend(ids[:args.limit_per_subject] if args.limit_per_subject else ids)
    if not selected:
        raise ValueError('No selected images')
    model = models.CellposeModel(pretrained_model=str(checkpoint), device=torch.device(args.device),
                                 use_bfloat16=False)
    partial, cache, flow_cache = [out/name for name in ('regions', 'maps', 'flows')]
    for folder in (partial, cache, flow_cache):
        folder.mkdir(exist_ok=True)
    modalities = ['invivo', 'exvivo'] if args.modality == 'both' else [args.modality]
    rows = []
    root = Path('data/raw/Project_2_Dataset')
    for sid in selected:
        saved = partial/f'{sid}.json'
        if saved.exists():
            rows.append(json.loads(saved.read_text())); continue
        row = copy.deepcopy(reference[sid]); row['match_pairs'] = []
        for modality in modalities:
            started = time.perf_counter()
            image = read_image(image_path(root, args.split, sid, modality))
            labels, flows, _ = model.eval(image, batch_size=1, diameter=None, min_size=15,
                                          flow_threshold=.4, cellprob_threshold=0., normalize=True)
            if labels.shape != image.shape or np.asarray(flows[2]).shape != image.shape:
                raise ValueError('Wrong inference dimensions')
            row[f'{modality}_instances'] = labels_to_instances(labels.astype(np.int32), 'CP4')
            np.savez_compressed(cache/f'{sid}_{modality}.npz',
                                maps=confidence_maps(labels, np.asarray(flows[2])).astype(np.float16))
            np.savez_compressed(flow_cache/f'{sid}_{modality}.npz',
                                dP=np.asarray(flows[1], dtype=np.float32),
                                cellprob=np.asarray(flows[2], dtype=np.float32), labels=labels.astype(np.int32))
            print('Cellpose 4', args.model, sid, modality, len(row[f'{modality}_instances']),
                  round(time.perf_counter()-started, 2), 'mps_driver_gb', round(torch.mps.driver_allocated_memory()/1e9,3) if args.device=='mps' else None, flush=True)
        saved.write_text(json.dumps(row)+'\n'); rows.append(row)
    write_rows(rows, out/'predictions.csv')
    (out/'completion.json').write_text(json.dumps({'regions': len(rows), 'sample_ids': selected,
                'complete_reference_coverage': set(selected) == set(reference),
                'csv_sha256': hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
