"""Direct EX center and log-SPD ellipse prediction, development pilot only."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from .center_ellipse import encode_points, transform_values
from .data import read_rows, read_image, image_path, split_id, parse_instances
from .learning import image_features, choose_device
from .resnet_ex import ResNetEX

ROOT = Path('data/raw/Project_2_Dataset')
OUT = Path('outputs/stage125_direct_center')
SUBJECTS = {'5d': 'subject_5d294c', 'db': 'subject_db6b8b'}
SEED = 2125


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_bank(bank, shape):
    keys = sorted(bank)
    if not keys:
        return np.empty((0, 2), int), np.empty((0, 5), np.float32)
    encoded = [encode_points(bank[k].coordinates()) for k in keys]
    rounded = np.array([a for a, _ in encoded])
    values = np.array([v for _, v in encoded])
    # Use unrounded centers for matching; offsets retain encode_points precision.
    centers = np.array([bank[k].coordinates().mean(0) for k in keys])
    options = np.array(sorted({(int(x+dx), int(y+dy)) for x, y in rounded
                               for dx in range(-2, 3) for dy in range(-2, 3)
                               if 0 <= x+dx < shape[1] and 0 <= y+dy < shape[0]}))
    cost = ((centers[:, None]-options[None])**2).sum(2)
    allowed = np.abs(rounded[:, None]-options[None]).max(2) <= 2
    cost[~allowed] = 1e12
    rows, cols = linear_sum_assignment(cost)
    assert np.array_equal(rows, np.arange(len(keys))) and (cost[rows, cols] < 1e12).all()
    anchors = options[cols]
    values[:, :2] += rounded-anchors
    assert np.abs(values[:, :2]).max() <= 2.5
    return anchors, values


def augment_geometry(anchors, values, size, turns, flip):
    anchors, values = anchors.copy(), values.copy()
    for _ in range(turns):
        rotation = np.array([[0, 1], [-1, 0]])
        anchors = anchors @ rotation.T + [0, size-1]
        values = np.array([transform_values(v, rotation) for v in values]).reshape(-1, 5)
    if flip:
        rotation = np.diag([-1, 1])
        anchors = anchors @ rotation.T + [size-1, 0]
        values = np.array([transform_values(v, rotation) for v in values]).reshape(-1, 5)
    return anchors, values.astype(np.float32)


def targets(anchors, values, size):
    heat = np.zeros((size, size), np.float32)
    regression = np.zeros((5, size, size), np.float32)
    valid = np.zeros((size, size), np.float32)
    for (x, y), v in zip(anchors, values):
        assert 0 <= x < size and 0 <= y < size and not valid[y, x]
        y0, y1, x0, x1 = max(y-3, 0), min(y+4, size), max(x-3, 0), min(x+4, size)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], np.exp(-((xx-x)**2+(yy-y)**2)/2.))
        regression[:, y, x] = v
        regression[2, y, x] -= 3.
        valid[y, x] = 1.
    return heat, regression, valid


def load_items(ids, truth):
    items = []
    for sid in ids:
        path = image_path(ROOT, 'training', sid, 'exvivo')
        image = read_image(path)
        anchors, values = encode_bank(parse_instances(truth[sid], 'exvivo', image.shape), image.shape)
        items.append(dict(sample_id=sid, x=image_features(image).astype(np.float16),
                          anchors=anchors, values=values))
    return items


def sample(items, rng, batch=4, patch=96):
    result = [[], [], [], []]
    for _ in range(batch):
        item = items[int(rng.integers(len(items)))]; _, h, w = item['x'].shape
        if len(item['anchors']) and rng.random() < .75:
            cx, cy = item['anchors'][int(rng.integers(len(item['anchors'])))] + rng.uniform(-patch/4, patch/4, 2)
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        x, y = int(np.clip(cx-patch/2, 0, max(0, w-patch))), int(np.clip(cy-patch/2, 0, max(0, h-patch)))
        image = item['x'][:, y:y+patch, x:x+patch].astype(np.float32)
        image = np.pad(image, ((0, 0), (0, patch-image.shape[1]), (0, patch-image.shape[2])))
        anchors = item['anchors']-[x, y]
        keep = ((anchors >= 0) & (anchors < patch)).all(1)
        anchors, values = anchors[keep], item['values'][keep]
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        image = np.rot90(image, turns, axes=(1, 2))
        if flip:
            image = image[:, :, ::-1]
        anchors, values = augment_geometry(anchors, values, patch, turns, flip)
        image = np.clip(image**rng.uniform(.7, 1.4)*rng.uniform(.8, 1.2)+rng.normal(0, .01, image.shape), 0, 1).astype(np.float32)
        heat, regression, valid = targets(anchors, values, patch)
        for dest, value in zip(result, [image, heat, regression, valid]):
            dest.append(value)
    return tuple(torch.from_numpy(np.stack(a)) for a in result)


def make_model(initial=None):
    model = ResNetEX()
    if initial is not None:
        model.load_state_dict(initial['state_dict'])
    model.out = nn.Conv2d(32, 6, 1)
    nn.init.normal_(model.out.weight, std=.001)
    nn.init.zeros_(model.out.bias)
    with torch.no_grad():
        model.out.bias[0] = -2.19
    return model


def loss_fn(logits, heat, regression, valid):
    probability = logits[:, 0].sigmoid().clamp(1e-5, 1-1e-5)
    count = valid.sum().clamp_min(1.)
    positive = -((1-probability)**2*probability.log()*valid).sum()/count
    negative = -(probability**2*(1-probability).log()*(1-heat)**4*(1-valid)).sum()/count
    error = F.smooth_l1_loss(logits[:, 1:], regression, reduction='none', beta=.1)
    offset = (error[:, :2]*valid[:, None]).sum()/(2*count)
    shape = (error[:, 2:]*valid[:, None]).sum()/(3*count)
    return positive+negative+2*offset+shape


def train(tag, steps, stop_after, device):
    truth = read_rows(ROOT/'training/train_ground_truth.csv')
    subject = SUBJECTS[tag]
    ids = sorted(s for s in truth if split_id(s)[0] == subject)
    initializer = Path(f'outputs/stage18_resnet_crossfit/train_{tag}/model.pt')
    initial = torch.load(initializer, map_location='cpu', weights_only=False)
    assert initial['training_subject'] == subject and set(initial['train_ids']) == set(ids)
    folder = OUT/f'train_{tag}'; folder.mkdir(parents=True, exist_ok=True)
    config = dict(training_subject=subject, train_ids=ids, steps=steps, seed=SEED,
                  batch=4, patch=96, encoder_lr=.00005, decoder_lr=.0003, head_lr=.001,
                  ema_decay=.99, initializer=str(initializer), initializer_sha256=sha(initializer),
                  annotation_sha256=hashlib.sha256(json.dumps({s: truth[s]['exvivo_instances'] for s in ids}, sort_keys=True).encode()).hexdigest(),
                  image_sha256={s: sha(image_path(ROOT, 'training', s, 'exvivo')) for s in ids},
                  sources={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('center_ellipse.py'), Path(__file__).with_name('resnet_ex.py'), Path(__file__).with_name('learning.py')]})
    path = folder/'configuration.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Changed configuration; preserve existing experiment.'
    else:
        path.write_text(json.dumps(config, indent=2)+'\n')
    final, resume = folder/'model.pt', folder/'resume.pt'
    if final.exists():
        saved = torch.load(final, map_location='cpu', weights_only=False)
        assert saved['config'] == config and saved['step'] == steps
        print('Completed model already exists.', flush=True)
        return
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model = make_model(initial).to(device); ema = copy.deepcopy(model).eval()
    encoder = list(model.encoder.parameters()); head = list(model.out.parameters())
    used = {id(p) for p in encoder+head}
    decoder = [p for p in model.parameters() if id(p) not in used]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': .00005}, {'params': decoder, 'lr': .0003}, {'params': head, 'lr': .001}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s+1)/50., 1.)*(.05+.95*(1+math.cos(math.pi*s/steps))/2))
    start = 0; history = []
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        assert saved['config'] == config
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state = saved['rng']; start = saved['step']; history = saved['history']
    items = load_items(ids, truth)
    model.train(); started = time.perf_counter()
    step = start
    for step in range(start+1, min(steps, stop_after or steps)+1):
        x, heat, regression, valid = (t.to(device) for t in sample(items, rng))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), heat, regression, valid)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.mul_(.99).add_(current, alpha=.01)
        if step == start+1 or step % 100 == 0 or step == min(steps, stop_after or steps):
            entry = dict(step=step, loss=float(loss.detach().cpu()), gradient_norm=float(norm.cpu()), seconds=time.perf_counter()-started)
            if device == 'mps':
                entry['allocated_gb'] = torch.mps.current_allocated_memory()/1e9
            history.append(entry); print(tag, entry, flush=True)
        if step % 250 == 0 or step == min(steps, stop_after or steps):
            temp = folder/'resume.tmp'
            torch.save(dict(config=config, step=step, model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng.bit_generator.state, history=history), temp)
            temp.replace(resume)
            (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    if step == steps:
        torch.save(dict(config=config, step=step, state_dict={k: v.cpu() for k, v in ema.state_dict().items()}), final)
    print('Training segment complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', choices=SUBJECTS, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    args = parser.parse_args()
    assert args.steps > 0 and (args.stop_after is None or 0 < args.stop_after <= args.steps)
    torch.set_num_threads(2)
    train(args.train, args.steps, args.stop_after, choose_device(args.device))
