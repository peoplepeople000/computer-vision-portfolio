"""Fixed-parameter public Cellpose baseline, separate from local training.

Inference reads images and existing prediction IDs only. Ground-truth masks and
verified pairs are not read. Evaluation is a separate operation after inference.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit
import torch

from .data import read_rows, write_rows, read_image, image_path, split_id
from .rle import labels_to_instances


def confidence_maps(labels, cell_logits):
    """Expose cell probability plus a mask-derived distance proxy, not a learned EDT."""
    center = np.zeros(labels.shape, np.float32)
    for n, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        distance = ndi.distance_transform_edt(np.pad(labels[box] == n, 1))[1:-1, 1:-1]
        center[box] = np.maximum(center[box], distance/max(float(distance.max()), 1.))
    return np.stack([expit(cell_logits), center]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['cyto3', 'nuclei'], default='cyto3')
    parser.add_argument('--diameter', type=float, default=10.)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--modality', choices=['invivo', 'exvivo', 'both'], default='exvivo')
    parser.add_argument('--limit-per-subject', type=int, default=0)
    parser.add_argument('--reference', default='outputs/stage9_subject_geometry/joint/predicted_all.csv')
    parser.add_argument('--tile-normalization', type=int, default=0)
    args = parser.parse_args()
    root = Path('data/raw/Project_2_Dataset')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(Path('models/cellpose').resolve())
    from cellpose import models
    torch.set_num_threads(2)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available')
    model = models.CellposeModel(model_type=args.model, device=torch.device(args.device))
    checkpoint = Path(model.pretrained_model)
    config = {**vars(args), 'cellpose_version': version('cellpose'), 'torch_version': torch.__version__,
              'pretrained_checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              'flow_threshold': .4, 'cellprob_threshold': 0., 'min_size': 12,
              'training': 'Public pretrained model; no competition-data fine-tuning.',
              'confidence_note': 'Second map is mask-derived normalized distance, not U-Net confidence. Recalibrate downstream classifiers before comparing full pipelines.'}
    config_path = out/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Different output configuration')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    reference = read_rows(args.reference)
    selected = []
    for subject in sorted({split_id(s)[0] for s in reference}):
        ids = sorted(s for s in reference if split_id(s)[0] == subject)
        selected.extend(ids[:args.limit_per_subject] if args.limit_per_subject else ids)
    modalities = ['invivo', 'exvivo'] if args.modality == 'both' else [args.modality]
    partial = out/'regions'; partial.mkdir(exist_ok=True)
    cache = out/'maps'; cache.mkdir(exist_ok=True)
    results = []
    for sid in selected:
        saved = partial/f'{sid}.json'
        if saved.exists():
            results.append(json.loads(saved.read_text()))
            continue
        row = copy.deepcopy(reference[sid]); row['match_pairs'] = []
        for modality in modalities:
            started = time.perf_counter()
            image = read_image(image_path(root, 'training', sid, modality))
            labels, flows, _ = model.eval(image, channels=[0, 0], diameter=args.diameter,
                                          batch_size=4, min_size=12, flow_threshold=.4,
                                          cellprob_threshold=0., normalize={'tile_norm_blocksize': args.tile_normalization})
            if labels.shape != image.shape or np.asarray(flows[2]).shape != image.shape:
                raise ValueError('Cellpose output shape mismatch')
            encoded = labels_to_instances(labels.astype(np.int32), 'IVP' if modality == 'invivo' else 'EXP')
            row[f'{modality}_instances'] = encoded
            np.savez_compressed(cache/f'{sid}_{modality}.npz',
                                maps=confidence_maps(labels, np.asarray(flows[2])).astype(np.float16))
            print('Cellpose', sid, modality, len(encoded), round(time.perf_counter()-started, 2), flush=True)
        saved.write_text(json.dumps(row)+'\n'); results.append(row)
    write_rows(results, out/'predictions.csv')
    (out/'completion.json').write_text(json.dumps({'regions': len(results), 'complete_47_region_validation': len(results) == 47,
                                                  'predictions_sha256': hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
