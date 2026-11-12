"""Twofold-resolution CPDINO EX adaptation with fixed budgets and excluded-mouse folds.

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
        image = read_image(image_path(root, 'training', sid, 'exvivo'))
        labels = instance_labels(truth[sid]['exvivo_instances'], image.shape)
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
                            x=np.pad(normalized, 40, mode='reflect'),
                            y=np.pad(flows, ((0, 0), (40, 40), (40, 40))),
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
            cy, cx = np.asarray([cy, cx]) + rng.uniform(-12, 12, 2)
        else:
            cy, cx = rng.uniform(0, record['shape'])
        y, x = np.clip(np.rint([cy, cx]).astype(int), [0, 0], np.array(record['shape']) - 1)
        crop = record['x'][y:y+80, x:x+80].copy()
        crop = np.maximum(crop, 0) ** rng.uniform(.8, 1.2) * rng.uniform(.85, 1.15)
        xs.append(np.stack([crop, np.zeros_like(crop), np.zeros_like(crop)]))
        ys.append(record['y'][:, y:y+80, x:x+80].copy())
    return random_rotate_and_resize(xs, lbls=ys, bsize=128, scale_range=.4, rescale=np.full(batch, .5, np.float32), device=device)[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', choices=[*FOLDS, 'all-training'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--infer-development', action='store_true')
    p.add_argument('--train-only', action='store_true')
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
    held_out = FOLDS.get(a.fold)
    ids = sorted(s for s in truth if split_id(s)[0] != held_out)
    out = a.output/a.fold
    out.mkdir(parents=True, exist_ok=True)
    initial = Path('models/cellpose4/cpdino-vitb')
    config = dict(modality='exvivo', inference_scale=2., held_out=held_out, train_ids=ids, steps=a.steps, seed=2073,
                  batch=2, train_patch=128, inference_patch=128, normalization='full image p1/p99',
                  sampling='75% cell centered, 25% random; balanced mice; 80-native-to-128 twofold official augmentation',
                  encoder_lr=1e-5, head_lr=1e-4, weight_decay=.1, rdrop=.1,
                  source_sha256=digest(__file__), initializer_sha256=digest(initial),
                  flow_target_version='Cellpose 4.2.1 labels_to_flows CPU',
                  loss='official flow MSE plus positive-weight-2 BCE and 0.3 soft Dice',
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
        if any(r['subject'] == held_out for r in records):
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
            target = (y[:, 0] > .5).to(prediction.dtype)
            flow_loss = torch.nn.functional.mse_loss(prediction[:, :2], 5. * y[:, 1:]) / 2.
            foreground = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[:, 2], target, pos_weight=torch.tensor(2., device=device))
            prob = prediction[:, 2].sigmoid()
            dice = 1. - (2. * (prob * target).sum((1, 2)) + 1.) / (prob.sum((1, 2)) + target.sum((1, 2)) + 1.)
            loss = flow_loss + foreground + .3 * dice.mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0:
                item = dict(step=step, loss=float(loss.detach().cpu()), seconds=time.perf_counter()-started)
                history.append(item)
                print('CPDINO 2x adaptation', a.fold, item, flush=True)
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
    if a.train_only:
        return
    model.net.eval()
    reference = read_rows('outputs/stage32_contour_consensus/predicted_all.csv')
    split = 'hidden_test' if held_out is None else 'training'
    if held_out is None:
        reference = read_rows('outputs/submission_stage32_cellpose_consensus/submission.csv')
    for name in ['regions', 'maps', 'flows']:
        (out/name).mkdir(exist_ok=True)
    rows = []
    for sid, base in reference.items():
        if held_out is not None and split_id(sid)[0] != held_out and not a.infer_development:
            continue
        path = out/'regions'/f'{sid}.json'
        if not path.exists():
            tick = time.perf_counter()
            image = read_image(image_path(root, split, sid, 'exvivo'))
            labels, flows, _ = model.eval(image, batch_size=2, diameter=15., bsize=128,
                                          flow_threshold=.4, cellprob_threshold=0., min_size=12)
            row = copy.deepcopy(base)
            row['exvivo_instances'] = labels_to_instances(labels.astype(np.int32), 'DINO')
            row['match_pairs'] = []
            np.savez_compressed(out/'maps'/f'{sid}_exvivo.npz',
                                maps=confidence_maps(labels, np.asarray(flows[2])).astype(np.float16))
            np.savez_compressed(out/'flows'/f'{sid}.npz', dP=np.asarray(flows[1], dtype=np.float32),
                                cellprob=np.asarray(flows[2], dtype=np.float32), labels=labels.astype(np.int32))
            path.write_text(json.dumps(row)+'\n')
            print('CPDINO adapted inference', sid, len(row['exvivo_instances']), round(time.perf_counter()-tick, 1), flush=True)
        row = json.loads(path.read_text())
        if held_out is None or split_id(sid)[0] == held_out:
            rows.append(row)
    write_rows(rows, out/'predictions.csv')
    (out/'completion.json').write_text(json.dumps(dict(checkpoint_sha256=digest(final),
        csv_sha256=digest(out/'predictions.csv'), regions=len(rows), train_ids=ids, held_out=held_out), indent=2)+'\n')
    if held_out is not None:
        evaluate(SimpleNamespace(data=root, predictions=out/'predictions.csv', subject=held_out, output=out/'metrics.json'))


if __name__ == '__main__':
    main()
