"""Native-resolution EX specialist with training-only difficult-positive sampling."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .learning import SmallUNet, load_training, choose_device, probability_maps, instances_from_maps
from .data import read_rows, parse_instances, read_image, image_path, split_id, write_rows
from .rle import labels_to_instances
from .stage3 import FOLDS, development_signature
from .cli import evaluate


def ex_loss(logits, target):
    fg = target[:, :1]; prob = logits.sigmoid()
    local = F.avg_pool2d(fg, 3, stride=1, padding=1)
    weight = 1 + ((local > 0) & (local < 1)).float()
    bce = F.binary_cross_entropy_with_logits(logits[:, :1], fg, reduction='none',
        pos_weight=torch.tensor(2., device=logits.device))
    bce = (bce * weight).sum() / weight.sum()
    dims = (1, 2, 3)
    dice = 1 - ((2*(prob[:, :1]*fg).sum(dims)+1)/(prob[:, :1].sum(dims)+fg.sum(dims)+1)).mean()
    distance = (F.smooth_l1_loss(prob[:, 1:], target[:, 1:], reduction='none')*(1+4*fg)).mean()
    return bce + dice + 2*distance


def difficult_centers(actual, predicted):
    return np.asarray([mask.coordinates().mean(0)[::-1] for mask in actual.values()
        if mask.area and max((mask.iou(p) for p in predicted.values()), default=0.) <= .75]).reshape(-1, 2)


def sample_ex(items, batch, patch, rng):
    if any(r['mode'] != 1 for r in items): raise ValueError('EX sampler received IV data')
    subjects = sorted({r['subject'] for r in items}); xs = []; ys = []
    for _ in range(batch):
        subject = subjects[int(rng.integers(len(subjects)))]
        candidates = [r for r in items if r['subject'] == subject]
        item = candidates[int(rng.integers(len(candidates)))]; h, w = item['x'].shape[1:]
        choice = rng.random()
        centers = item['difficult_centers'] if choice < .3 and len(item['difficult_centers']) else item['centers']
        if choice < .8 and len(centers):
            cy, cx = centers[int(rng.integers(len(centers)))] + rng.uniform(-patch/3, patch/3, 2)
        else: cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        y = int(np.clip(cy-patch/2, 0, max(0,h-patch))); x = int(np.clip(cx-patch/2, 0, max(0,w-patch)))
        feature = item['x'][:,y:y+patch,x:x+patch].astype(np.float32)
        target = item['y'][:,y:y+patch,x:x+patch].astype(np.float32)
        pad = ((0,0),(0,patch-feature.shape[1]),(0,patch-feature.shape[2]))
        feature = np.pad(feature,pad,mode='reflect'); target = np.pad(target,pad)
        k = int(rng.integers(4)); feature = np.rot90(feature,k,(1,2)); target = np.rot90(target,k,(1,2))
        if rng.random() < .5: feature = feature[:,:,::-1]; target = target[:,:,::-1]
        feature = np.clip(feature**rng.uniform(.7,1.4)*rng.uniform(.8,1.2)+rng.normal(0,.01,feature.shape),0,1).astype(np.float32)
        xs.append(np.concatenate([feature,np.ones((1,patch,patch),np.float32)])); ys.append(target)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def train_fold(args, tag):
    root = Path(args.data); out = Path(args.output); heldout = FOLDS[tag]
    folder = out/f'unet_holdout_{tag}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'model.pt').exists(): raise FileExistsError('Existing specialist weights are protected')
    initializer = Path(f'outputs/stage2/unet_holdout_{tag}/model.pt')
    state = torch.load(initializer, map_location='cpu', weights_only=False)
    if state['held_out'] != heldout: raise ValueError('Wrong initializer fold')
    all_items, train_ids = load_training(root, heldout)
    items = [r for r in all_items if r['mode'] == 1]; del all_items
    if set(train_ids) != set(state['train_ids']): raise ValueError('Wrong initializer training IDs')
    dev = Path(f'outputs/stage3/{tag}')
    if json.loads((dev/'development_manifest.json').read_text()) != development_signature(initializer):
        raise ValueError('Stale development provenance')
    predictions = read_rows(dev/'development.csv'); truth = read_rows(root/'training/train_ground_truth.csv')
    if set(predictions) != set(train_ids): raise ValueError('Development contains held-out data')
    for item in items:
        sid = item['sample_id']; shape = item['y'].shape[1:]
        actual = parse_instances(truth[sid], 'exvivo', shape)
        predicted = parse_instances(predictions[sid], 'exvivo', shape)
        item['difficult_centers'] = difficult_centers(actual, predicted)
    device = choose_device(args.device); torch.manual_seed(2028); rng = np.random.default_rng(2028)
    model = SmallUNet(state['width']).to(device); model.load_state_dict(state['state_dict']); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00003)
    config = {'held_out': heldout, 'train_ids': train_ids, 'steps': args.steps, 'seed': 2028,
        'device': device, 'initializer_sha256': hashlib.sha256(initializer.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sampling': 'EX only; balanced subject; 30% difficult positive, 50% any positive, 20% random; fallback to any positive',
        'boundary_weight': 2, 'positive_bce_weight': 2, 'spatial_scale': 'native; no resampling',
        'difficult_centers': sum(len(r['difficult_centers']) for r in items)}
    (folder/'training_config.json').write_text(json.dumps(config, indent=2)+'\n')
    start = time.perf_counter(); history = []
    for step in range(1, args.steps+1):
        x,y = sample_ex(items,8,128,rng); optimizer.zero_grad(set_to_none=True)
        loss = ex_loss(model(x.to(device)),y.to(device)); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); scheduler.step()
        value = float(loss.detach().cpu())
        if not np.isfinite(value): raise RuntimeError('Nonfinite loss')
        history.append({'step': step, 'loss': value, 'seconds': time.perf_counter()-start})
        if step == 1 or step % 200 == 0: print(tag,history[-1],flush=True)
    checkpoint = {'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
        'width': state['width'], 'step': state['step']+args.steps, 'held_out': heldout,
        'train_ids': train_ids, 'config': config}
    torch.save(checkpoint,folder/'model.pt'); (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    del items; model.eval()
    prior = read_rows('outputs/stage5_hybrid_geometry/predicted_all.csv'); rows = []
    cache = out/f'unet_holdout_{tag}_maps'; cache.mkdir(exist_ok=True)
    for sid in truth:
        if split_id(sid)[0] != heldout: continue
        image = read_image(image_path(root,'training',sid,'exvivo'))
        maps = probability_maps(model,image,1,device)
        np.savez_compressed(cache/f'{sid}_exvivo.npz', maps=maps.astype(np.float16))
        rows.append({'sample_id':sid, 'invivo_instances':prior[sid]['invivo_instances'],
            'exvivo_instances':labels_to_instances(instances_from_maps(maps,.5,.35),'EXP'), 'match_pairs':[]})
        print('EX prediction',sid,flush=True)
    path = out/f'unet_holdout_{tag}.csv'; write_rows(rows,path)
    Path(str(path)+'.manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=heldout,output=out/f'unet_holdout_{tag}_metrics.json'))
    del model
    if device == 'mps': torch.mps.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/raw/Project_2_Dataset'); p.add_argument('--output',default='outputs/stage6_ex')
    p.add_argument('--device',default='mps'); p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--fold',choices=['all',*FOLDS],default='all'); args=p.parse_args()
    torch.set_num_threads(2)
    for tag in (FOLDS if args.fold=='all' else [args.fold]): train_fold(args,tag)
    if args.fold=='all':
        rows=[]
        for tag in FOLDS: rows.extend(read_rows(Path(args.output)/f'unet_holdout_{tag}.csv').values())
        path=Path(args.output)/'segmentation_oof.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=Path(args.data),predictions=path,subject=None,output=Path(args.output)/'segmentation_oof_metrics.json'))


if __name__ == '__main__': main()
