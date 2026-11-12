"""Native-scale IV CPDINO adaptation with fixed budgets and excluded-mouse folds.

Run with PYTHONPATH=vendor/cellpose4:src in .venv-cellpose. Training crops
use full-image normalization, official flow targets and official geometric
augmentation. No held-out masks are used for training or checkpoint selection.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import read_rows, read_image, image_path, split_id, write_rows
from .cellpose_finetune import instance_labels
from .cellpose_baseline import confidence_maps
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cli import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training(root, truth, ids, cache):
    from cellpose import dynamics, transforms
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in ids:
        image = read_image(image_path(root, 'training', sid, 'invivo'))
        labels = instance_labels(truth[sid]['invivo_instances'], image.shape)
        checksum = hashlib.sha256(labels.tobytes()).hexdigest()
        target_path = cache / f'{sid}_{checksum[:16]}.npz'
        if not target_path.exists():
            flows = dynamics.labels_to_flows([labels.copy()], device=torch.device('cpu'))[0][1:]
            np.savez_compressed(target_path, flows=flows.astype(np.float32), checksum=checksum)
        with np.load(target_path) as saved:
            if saved['checksum'].item() != checksum:
                raise ValueError('Flow cache does not match training annotations')
            flows = saved['flows'].copy()
        normalized = transforms.normalize99(image.astype(np.float32))
        centers = np.asarray(ndi.center_of_mass(labels > 0, labels, np.unique(labels[labels > 0])))
        records.append(dict(sample_id=sid, subject=split_id(sid)[0],
                            x=np.pad(normalized, 80, mode='reflect'),
                            y=np.pad(flows, ((0, 0), (80, 80), (80, 80))),
                            shape=image.shape, centers=centers))
    return records


def sample(records, rng, device, batch=2):
    from cellpose.transforms import random_rotate_and_resize
    subjects = sorted({r['subject'] for r in records})
    xs, ys = [], []
    for _ in range(batch):
        subject = subjects[int(rng.integers(len(subjects)))]
        group = [r for r in records if r['subject'] == subject]
        record = group[int(rng.integers(len(group)))]
        if rng.random() < .75 and len(record['centers']):
            cy, cx = record['centers'][int(rng.integers(len(record['centers'])))]
            cy, cx = np.asarray([cy, cx]) + rng.uniform(-24, 24, 2)
        else:
            cy, cx = rng.uniform(0, record['shape'])
        y, x = np.clip(np.rint([cy, cx]).astype(int), [0, 0], np.array(record['shape']) - 1)
        crop = record['x'][y:y+160, x:x+160].copy()
        crop = np.maximum(crop, 0) ** rng.uniform(.8, 1.2) * rng.uniform(.85, 1.15)
        xs.append(np.stack([crop, np.zeros_like(crop), np.zeros_like(crop)]))
        ys.append(record['y'][:, y:y+160, x:x+160].copy())
    return random_rotate_and_resize(xs, lbls=ys, bsize=128, scale_range=.4, device=device)[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-subject', choices=list(FOLDS.values()), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    a = p.parse_args()
    from cellpose import models
    from cellpose.train import _loss_fn_seg
    torch.set_num_threads(2)
    torch.manual_seed(2073)
    np.random.seed(2073)
    rng = np.random.default_rng(2073)
    device = torch.device(a.device)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    excluded_subjects = sorted(set(FOLDS.values()) - {a.train_subject})
    ids = sorted(s for s in truth if split_id(s)[0] == a.train_subject)
    out = a.output/a.train_subject
    out.mkdir(parents=True, exist_ok=True)
    initial = Path('models/cellpose4/cpdino-vitb')
    config = dict(modality='invivo', training_subject=a.train_subject, excluded_subjects=excluded_subjects, train_ids=ids, steps=a.steps, seed=2073,
                  batch=2, train_patch=128, inference_patch=128, normalization='full image p1/p99',
                  sampling='75% cell centered, 25% random; balanced mice; 160-to-128 official augmentation',
                  encoder_lr=1e-5, head_lr=1e-4, weight_decay=.1, rdrop=.1,
                  source_sha256=digest(__file__), initializer_sha256=digest(initial),
                  flow_target_version='Cellpose 4.2.1 labels_to_flows CPU',
                  annotation_sha256=digest(root/'training/train_ground_truth.csv'))
    cfg = out/'configuration.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Configuration changed; choose a fresh output directory')
    cfg.write_text(json.dumps(config, indent=2)+'\n')
    model = models.CellposeModel(pretrained_model=str(initial), device=device, use_bfloat16=False)
    final, resume = out/'model.pt', out/'resume.pt'
    if final.exists():
        saved = torch.load(final, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['step'] != a.steps:
            raise ValueError('Checkpoint provenance mismatch')
        model.net.load_state_dict(saved['state_dict'])
    else:
        records = load_training(root, truth, ids, a.output/'flow_targets')
        if any(r['subject'] != a.train_subject for r in records):
            raise ValueError('Evaluation mouse entered training')
        optimizer = torch.optim.AdamW([
            {'params': model.net.encoder.parameters(), 'lr': 1e-5},
            {'params': model.net.out.parameters(), 'lr': 1e-4}], weight_decay=.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.steps, eta_min=1e-6)
        start, history = 0, []
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=False)
            if saved['config'] != config:
                raise ValueError('Resume provenance mismatch')
            model.net.load_state_dict(saved['state_dict'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            start, history = saved['step'], saved['history']
            rng.bit_generator.state = saved['rng']
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if a.device == 'mps':
                torch.mps.set_rng_state(saved['mps_rng'])
        model.net.rdrop = .1
        model.net.train()
        started = time.perf_counter()
        for step in range(start+1, a.steps+1):
            x, y = sample(records, rng, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.net(x)[0]
            loss = _loss_fn_seg(y, prediction, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0:
                item = dict(step=step, loss=float(loss.detach().cpu()), seconds=time.perf_counter()-started)
                history.append(item)
                print('Single-mouse CPDINO IV adaptation', a.train_subject, item, flush=True)
            if step % 200 == 0 or step == a.steps:
                state = dict(config=config, step=step, state_dict=model.net.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             rng=rng.bit_generator.state, numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(),
                             mps_rng=torch.mps.get_rng_state() if a.device == 'mps' else None,
                             history=history)
                torch.save(state, out/'resume.tmp')
                (out/'resume.tmp').replace(resume)
        torch.save(dict(config=config, step=a.steps,
                        state_dict={k:v.detach().cpu() for k,v in model.net.state_dict().items()}), final)
        (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        del records, optimizer, scheduler, state, x, y, prediction, loss
    return


if __name__ == '__main__':
    main()
