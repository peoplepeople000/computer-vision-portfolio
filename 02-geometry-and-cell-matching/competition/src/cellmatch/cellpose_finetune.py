"""Outer-fold Cellpose adaptation with a fixed budget and no held-out training.

Use the isolated .venv-cellpose environment. This script does not select
checkpoints using validation scores and does not upload data.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch

from .data import read_rows, read_image, image_path, split_id, write_rows
from .rle import Mask, labels_to_instances
from .cellpose_baseline import confidence_maps

FOLDS = {'5d': 'subject_5d294c', 'b2': 'subject_b2ba5e', 'db': 'subject_db6b8b'}


def instance_labels(encoded, shape):
    labels = np.zeros(shape, np.int32)
    for n, rle in enumerate(encoded.values(), 1):
        mask = Mask.parse(rle, shape)
        for start, end in zip(mask.starts, mask.ends):
            if labels.ravel()[start:end].any():
                raise ValueError('Training instances overlap')
            labels.ravel()[start:end] = n
    return labels


def centered_patches(image, labels, rng, size=72, count=32):
    """Extract training-only cell-centered crops, retaining surrounding background."""
    if size < 16 or size % 2 or count < 1:
        raise ValueError('Expected a positive count and an even patch size of at least 16')
    ids = np.unique(labels); ids = ids[ids > 0]
    if not len(ids):
        raise ValueError('Cell-centered training needs annotated instances')
    selected = rng.choice(ids, size=min(count, len(ids)), replace=False)
    half = size//2
    padded_image = np.pad(image, half, mode='reflect')
    padded_labels = np.pad(labels, half, mode='constant')
    result = []
    for key in selected:
        y, x = np.rint(np.argwhere(labels == key).mean(0)).astype(int)
        patch = padded_image[y:y+size, x:x+size].copy()
        target = padded_labels[y:y+size, x:x+size].copy()
        # Cellpose expects consecutive integer instance IDs with background zero.
        for n, old in enumerate(np.unique(target[target > 0]), 1):
            target[padded_labels[y:y+size, x:x+size] == old] = n
        result.append((patch, target))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=['all', *FOLDS], default='all')
    parser.add_argument('--output', default='outputs/stage12_cellpose_adapted')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--infer-development', action='store_true')
    parser.add_argument('--sampling', choices=['whole-image', 'centered'], default='whole-image')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--patch-size', type=int, default=72)
    parser.add_argument('--patches-per-image', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(Path('models/cellpose').resolve())
    from cellpose import models, train, transforms
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    reference = read_rows('outputs/stage9_subject_geometry/joint/predicted_all.csv')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for tag in (FOLDS if args.fold == 'all' else [args.fold]):
        subject = FOLDS[tag]
        ids = sorted(s for s in truth if split_id(s)[0] != subject)
        folder = out/tag; folder.mkdir(exist_ok=True)
        config = {'held_out': subject, 'train_ids': ids, 'epochs': args.epochs,
                  'seed': 2051, 'initial_model': 'cyto3', 'batch_size': 8, 'nimg_per_epoch': 32,
                  'learning_rate': args.learning_rate, 'weight_decay': .01, 'optimizer': 'AdamW',
                  'sampling': args.sampling, 'patch_size': args.patch_size,
                  'patches_per_image': args.patches_per_image,
                  'normalization': 'Full-image Cellpose normalization before centered crop extraction' if args.sampling == 'centered' else 'Cellpose full-image normalization',
                  'cellpose_version': version('cellpose'), 'device': args.device,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'policy': 'Fixed final epoch; no held-out images, masks, or pairs enter training.'}
        config_path = folder/'training_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Changed training configuration')
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        complete = folder/'training_complete.json'
        if complete.exists():
            finished = json.loads(complete.read_text())
            checkpoint = Path(finished['checkpoint'])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != finished['sha256']:
                raise ValueError('Trained checkpoint changed')
            diameter = finished['diameter']
        else:
            if (folder/'models').exists() and list((folder/'models').iterdir()):
                raise FileExistsError('Partial checkpoints retained; verify old process and choose a new output before restarting')
            np.random.seed(2051); torch.manual_seed(2051)
            model = models.CellposeModel(model_type='cyto3', device=torch.device(args.device))
            images, labels, subjects = [], [], []
            rng = np.random.default_rng(2051)
            seen = set()
            for sid in ids:
                image = read_image(image_path(root, 'training', sid, 'exvivo'))
                target = instance_labels(truth[sid]['exvivo_instances'], image.shape)
                key = hashlib.sha256(image.tobytes()+target.tobytes()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                if args.sampling == 'centered':
                    normalized = transforms.normalize99(image.astype(np.float32))
                    samples = centered_patches(normalized, target, rng, args.patch_size, args.patches_per_image)
                else:
                    samples = [(image, target)]
                for sample, annotation in samples:
                    images.append(sample); labels.append(annotation); subjects.append(split_id(sid)[0])
            if subject in subjects or len(set(subjects)) != 2:
                raise ValueError('Invalid outer-fold training subjects')
            counts = {s: subjects.count(s) for s in set(subjects)}
            probabilities = np.array([1./counts[s] for s in subjects]); probabilities /= probabilities.sum()
            (folder/'initializer.json').write_text(json.dumps({
                'path': str(model.pretrained_model), 'sha256': hashlib.sha256(Path(model.pretrained_model).read_bytes()).hexdigest(),
                'training_images': len(images), 'subject_counts': counts}, indent=2)+'\n')
            checkpoint, losses, _ = train.train_seg(
                model.net, train_data=images, train_labels=labels, train_probs=probabilities,
                test_data=None, test_labels=None, channels=[0, 0], normalize=args.sampling != 'centered',
                batch_size=8, learning_rate=args.learning_rate, weight_decay=.01, SGD=False,
                n_epochs=args.epochs, nimg_per_epoch=32, rescale=True, min_train_masks=1,
                save_path=str(folder.resolve()), save_every=50, model_name=f'ex_{tag}_final')
            checkpoint = Path(checkpoint)
            diameter = float(model.net.diam_labels.detach().cpu().item())
            complete.write_text(json.dumps({'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                           'diameter': diameter, 'completed_epochs': args.epochs}, indent=2)+'\n')
            (folder/'losses.json').write_text(json.dumps(np.asarray(losses).tolist())+'\n')
            del model, images, labels
            if args.device == 'mps':
                torch.mps.empty_cache()
        model = models.CellposeModel(pretrained_model=str(checkpoint), device=torch.device(args.device))
        cache = folder/'maps'; cache.mkdir(exist_ok=True)
        partial = folder/'regions'; partial.mkdir(exist_ok=True)
        rows = []
        for sid in truth:
            heldout = split_id(sid)[0] == subject
            if not heldout and not args.infer_development:
                continue
            saved = partial/f'{sid}.json'
            if saved.exists():
                row = json.loads(saved.read_text())
            else:
                image = read_image(image_path(root, 'training', sid, 'exvivo'))
                labels, flows, _ = model.eval(image, channels=[0, 0], diameter=diameter, batch_size=4,
                                              min_size=12, flow_threshold=.4, cellprob_threshold=0.)
                row = dict(reference[sid]); row['match_pairs'] = []
                row['exvivo_instances'] = labels_to_instances(labels.astype(np.int32), 'EXP')
                np.savez_compressed(cache/f'{sid}_exvivo.npz', maps=confidence_maps(labels, np.asarray(flows[2])).astype(np.float16))
                saved.write_text(json.dumps(row)+'\n')
                print('Adapted Cellpose', tag, sid, len(row['exvivo_instances']), flush=True)
            if heldout:
                rows.append(row)
        write_rows(rows, folder/'heldout.csv')
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
    if args.fold == 'all':
        rows = [r for tag in FOLDS for r in read_rows(out/tag/'heldout.csv').values()]
        write_rows(rows, out/'segmentation_oof.csv')


if __name__ == '__main__':
    main()
