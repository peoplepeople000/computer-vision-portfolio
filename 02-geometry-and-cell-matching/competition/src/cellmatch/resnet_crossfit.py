"""Single-mouse ResNet models for out-of-sample development segmentation.

For outer mouse C, train A -> predict B and train B -> predict A. No C image or
annotation is used in fitting or development calibration. Models are reusable
across outer folds because each initializer/training recipe depends only on A.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch

from .resnet_ex import ResNetEX, WEIGHTS, spatial_augment
from .learning import image_features, target_maps, sample_batch, probability_maps, choose_device
from .ex_specialist import ex_loss
from .data import read_rows, read_image, image_path, split_id
from .stage3 import FOLDS


def load_subset(root, truth, ids):
    items = []; seen = set()
    for sid in ids:
        image = read_image(image_path(root, 'training', sid, 'exvivo'))
        encoded = truth[sid]['exvivo_instances']
        key = hashlib.sha256(image.tobytes()+json.dumps(encoded, sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        target, labels = target_maps(encoded, image.shape)
        indices = np.unique(labels); indices = indices[indices > 0]
        centers = np.asarray(ndi.center_of_mass(labels > 0, labels, indices)).reshape(-1, 2)
        items.append({'x': image_features(image).astype(np.float16), 'y': target.astype(np.float16),
                      'centers': centers, 'mode': 1, 'subject': split_id(sid)[0], 'sample_id': sid})
    return items


def train_single(folder, subject, truth, root, steps, device):
    ids = sorted(s for s in truth if split_id(s)[0] == subject)
    if not ids:
        raise ValueError('Empty training mouse')
    config = {'training_subject': subject, 'train_ids': ids, 'steps': steps, 'seed': 2061,
              'batch': 4, 'patch': 96, 'encoder_lr': .0001, 'decoder_lr': .001, 'ema_decay': .99,
              'initializer_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'architecture_sha256': hashlib.sha256(Path(__file__).with_name('resnet_ex.py').read_bytes()).hexdigest(),
              'learning_sha256': hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest()}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'training_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Changed single-mouse training configuration')
    path.write_text(json.dumps(config, indent=2)+'\n')
    final, resume = folder/'model.pt', folder/'resume.pt'
    torch.manual_seed(2061); rng = np.random.default_rng(2061)
    model = ResNetEX(None if final.exists() or resume.exists() else WEIGHTS).to(device)
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        if state['config'] != config or state['step'] != steps:
            raise ValueError('Invalid completed single-mouse model')
        model.load_state_dict(state['state_dict']); return model.eval(), config
    items = load_subset(root, truth, ids)
    if any(r['subject'] != subject or r['sample_id'] not in ids for r in items):
        raise ValueError('Unexpected training mouse')
    ema = copy.deepcopy(model).eval()
    encoder = list(model.encoder.parameters()); encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': .0001}, {'params': decoder, 'lr': .001}], weight_decay=1e-4)
    def multiplier(step):
        return min((step+1)/100., 1.)*(.05+.95*(1+math.cos(math.pi*step/steps))/2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start, history = 0, []
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state['config'] != config:
            raise ValueError('Changed resume configuration')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['rng']; start = state['step']; history = state['history']
    model.train(); started = time.perf_counter()
    for step in range(start+1, steps+1):
        x, y = sample_batch(items, 4, 96, rng); x, y = spatial_augment(x, y, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = ex_loss(model(x.to(device)), y.to(device)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.mul_(.99).add_(current, alpha=.01)
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite single-mouse loss')
        if step == 1 or step % 100 == 0:
            entry = {'step': step, 'loss': value, 'seconds': time.perf_counter()-started}
            history.append(entry); print('Crossfit ResNet', subject, entry, flush=True)
        if step % 500 == 0 or step == steps:
            temporary = folder/'resume.tmp'
            torch.save({'config': config, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng': rng.bit_generator.state, 'history': history}, temporary)
            temporary.replace(resume)
    torch.save({'config': config, 'step': steps, 'training_subject': subject, 'train_ids': ids,
                'state_dict': {k: v.detach().cpu() for k, v in ema.state_dict().items()}}, final)
    (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    return ema.eval(), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', choices=list(FOLDS), required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage18_resnet_crossfit'))
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2); device = choose_device(args.device)
    root = Path('data/raw/Project_2_Dataset'); truth = read_rows(root/'training/train_ground_truth.csv')
    outer = FOLDS[args.outer]
    development = [tag for tag in FOLDS if tag != args.outer]
    manifest = {'held_out': outer, 'train_ids': sorted(s for s in truth if split_id(s)[0] != outer), 'sources': []}
    for tag in development:
        subject = FOLDS[tag]
        prediction_subject = FOLDS[next(t for t in development if t != tag)]
        ids = sorted(s for s in truth if split_id(s)[0] == prediction_subject)
        folder = args.output/f'train_{tag}'
        model, config = train_single(folder, subject, truth, root, args.steps, device)
        if outer == subject or outer == prediction_subject or set(ids) & set(config['train_ids']):
            raise ValueError('Outer or calibration-image training leakage')
        cache = folder/'maps'; cache.mkdir(exist_ok=True)
        for sid in ids:
            path = cache/f'{sid}_exvivo.npz'
            if path.exists():
                continue
            image = read_image(image_path(root, 'training', sid, 'exvivo'))
            maps = probability_maps(model, image, 1, device, tile=96, overlap=48, batch_size=4)
            np.savez_compressed(path, maps=maps.astype(np.float16))
            print('Crossfit development', subject, sid, flush=True)
        manifest['sources'].append({'training_subject': subject, 'prediction_subject': prediction_subject,
            'train_ids': config['train_ids'], 'prediction_ids': ids, 'folder': str(folder),
            'checkpoint_sha256': hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),
            'map_sha256': {s: hashlib.sha256((cache/f'{s}_exvivo.npz').read_bytes()).hexdigest() for s in ids}})
        del model
        if device == 'mps':
            torch.mps.empty_cache()
    (args.output/f'outer_{args.outer}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
