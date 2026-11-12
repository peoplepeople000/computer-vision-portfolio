"""Refit the selected architecture on public data only, with fixed schedules.

Training consumes the frozen numerical perception cache. RGB LoRA additionally
regenerates native crops from external videos and runs the pretrained encoder.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from run import HERE, SEEDS, sha
from sam_core import dump


def validate(features):
    a = dict(np.load(features / 'public_trees.npz', allow_pickle=False))
    sam = dict(np.load(features / 'sam_public.npz', allow_pickle=False))
    assert a['labels'].shape == (1299, 2) and len(set(a['clips'])) == 1299
    assert '0526_public.mp4' not in a['clips']
    assert np.isin(a['labels'], [0, 1, 2]).all()
    assert set(a['folds']) == {0, 1, 2, 3, 4}
    np.testing.assert_array_equal(a['clips'], sam['clips'])
    for key in ['fold0', 'fold2']:
        assert a[key].shape == (2598, 435)
    assert a['keeper_late3'].shape == (2598, 888)
    for source in set(a['sources']):
        assert len(set(a['folds'][a['sources'] == source])) == 1
    return a, sam


def prior_table(foot, y):
    y = torch.tensor(y, dtype=torch.int64)
    global_count = torch.bincount(y, minlength=3).float() + 1
    table = []
    for f in range(3):
        ids = np.flatnonzero(foot.argmax(1) == f)
        count = torch.bincount(y[ids], minlength=3).float() + 1 if len(ids) else global_count
        table.append((count/count.sum()).log())
    return torch.stack(table).numpy()


def train_trees(a, models, deadline):
    from catboost import CatBoostClassifier, Pool
    priors = {k: prior_table(a['fold0'][:1299, 186:189], a['labels'][:, t])
              for t, k in enumerate(['kick', 'keeper'])}
    np.savez_compressed(models / 'prior_tables.npz', **priors)
    jobs = [(f'{bank}_{level}', bank, width, 0) for bank in ['fold0', 'fold2']
            for level, width in [('2d', 189), ('body', 312), ('field', 435)]]
    jobs += [('control', 'keeper_static', 189, 1), ('keeper', 'keeper_late3', 888, 1)]
    for name, key, width, target in jobs:
        x = a[key][:, :width]; y = a['labels'][:, target]
        y = np.r_[y, 2-y]
        prior = priors['kick' if target == 0 else 'keeper']
        original = prior[a[key][:1299, 186:189].argmax(1)] if target == 0 else prior[a['keeper_static'][:1299, 186:189].argmax(1)]
        baseline = np.r_[original, original[:, [2, 1, 0]]]
        for seed in SEEDS:
            folder = models / f'trees/{name}/{seed}'; folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'model.cbm'
            if path.exists():
                audit = json.loads((folder / 'fit.json').read_text())
                assert sha(path) == audit['sha256']
                continue
            if time.monotonic() > deadline - 120:
                raise SystemExit('Budget boundary: completed tree models saved; rerun the same command to resume')
            model = CatBoostClassifier(iterations=450, depth=4, learning_rate=.035, l2_leaf_reg=15,
                loss_function='MultiClass', classes_count=3, random_seed=seed, task_type='GPU',
                devices='0', use_best_model=False, verbose=False, thread_count=4,
                allow_writing_files=True, train_dir=str(folder / 'training'), save_snapshot=True,
                snapshot_file=str((folder / 'training/progress.cbsnapshot').resolve()), snapshot_interval=30)
            model.fit(Pool(x, label=y, weight=np.full(2598, .5, np.float32), baseline=baseline))
            assert model.tree_count_ == 450
            model.save_model(str(path))
            dump(folder / 'fit.json', dict(sha256=sha(path), seed=seed, public=1299,
                                           augmented=2598, validation_used=False, trees=450))
            print(json.dumps(dict(stage='tree_complete', branch=name, seed=seed)), flush=True)


def train_sam(features, models, work, deadline):
    import sam_training
    a = dict(np.load(features / 'sam_public.npz', allow_pickle=False))
    rows = [r for r in json.loads((HERE / 'assets/observations.json').read_text()) if r['population'] == 'public']
    np.testing.assert_array_equal(a['clips'], [r['clip'] for r in rows])
    labels = np.array([r['label'] for r in rows], int)
    data = [torch.tensor(a[k], device='cuda') for k in ['tokens', 'mask', 'times', 'axis']]
    protocol_sha = sha(work / 'training_protocol.json')
    sam_training.fit(features, work / 'sam_fit', 'full', 36, data, rows, labels, protocol_sha, deadline-60)
    for seed in SEEDS:
        ck = torch.load(work / f'sam_fit/full/{seed}/epoch36.pt', map_location='cpu', weights_only=False)
        folder = models / 'sam'; folder.mkdir(exist_ok=True)
        torch.save({'head': ck['head']}, folder / f'{seed}.pt')


def train_rgb(args, models, work, deadline):
    import body_visual_probe as visual
    import human_rgb_adaptation as adapt
    import formal_training as training
    from encode import crops, load_rows
    from PIL import Image
    if args.dataset is None or args.pretrained is None:
        raise ValueError('RGB refit requires --dataset and --pretrained')
    rows = json.loads((HERE / 'assets/rgb_samples.json').read_text())[:1299]
    observations = {r['clip']: r for r in load_rows('public')}
    weights = args.pretrained / 'model.ckpt'
    contract = json.loads((HERE / 'assets/model_contract.json').read_text())['base_pretrained']
    assert sha(weights) == contract['weights_sha256']
    model, _ = visual.encoder(weights, 'sam_body_dinov3', contract['dinov3_revision'])
    model.half().eval(); base_digest = adapt.base_hash(model)
    prefix_dir = work / 'rgb_prefix'; prefix_dir.mkdir(exist_ok=True)
    hw = (24, 24); rope = model.rope_embed(H=24, W=24)
    for row in rows:
        if not row['available']:
            continue
        path = prefix_dir / (row['clip'] + '.npy')
        record_path = path.with_suffix('.json')
        identity = dict(video_sha256=observations[row['clip']]['video_sha256'], base=base_digest,
                        flip=row['flip'], last_frame=observations[row['clip']]['frames'][-1])
        if path.exists():
            old = json.loads(record_path.read_text())
            assert old['input'] == identity and sha(path) == old['sha256']
            continue
        if time.monotonic() > deadline - 180:
            raise SystemExit('Budget boundary: prefix caches saved; rerun the same command to resume')
        observed = dict(observations[row['clip']]); observed['frames'] = [dict(f, available=f['available'] and t == 4) for t, f in enumerate(observed['frames'])]
        crops(args.dataset, work / 'native_pixels', [observed])
        image = np.asarray(Image.open(work / 'native_pixels' / observed['frames'][-1]['file']).convert('RGB')).copy()
        if row['flip']:
            image = image[:, ::-1].copy()
        x = torch.from_numpy(image).permute(2, 0, 1).float()/255
        x = (x-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        with torch.inference_mode():
            tokens, shape, _ = adapt.prefix(model, x[None].cuda().half(), 24)
        assert tuple(shape) == hw and tokens.is_cuda
        np.save(path, tokens[0].cpu().numpy())
        dump(record_path, dict(input=identity, sha256=sha(path)))
    adapt.inject(model, 8, 8, 8)
    labels = np.load(args.features / 'rgb_training.npz')['canonical_labels']
    def feature(i, arm, fitting):
        if not rows[i]['available']:
            return torch.zeros(1, 5120, device='cuda')
        z = torch.from_numpy(np.load(prefix_dir / (rows[i]['clip'] + '.npy')).copy())[None].cuda()
        with torch.autocast('cuda', dtype=torch.float16, enabled=fitting):
            return adapt.suffix(model, z, hw, rope, 24).float()
    for seed in SEEDS:
        training.reset_adapters(model, seed); torch.manual_seed(seed)
        initial = {k: v.clone() for k, v in adapt.DirectionHead().state_dict().items()}
        head = training.fit(model, initial, seed, 'adapted', rows, labels, feature,
                            work / f'rgb_fit/{seed}', sha(work / 'training_protocol.json'),
                            base_digest, lambda: time.monotonic() > deadline-120)
        if head is None:
            raise SystemExit('Budget boundary: optimizer and RGB checkpoint saved for continuation')
        state = torch.load(work / f'rgb_fit/{seed}/checkpoint.pt', map_location='cpu', weights_only=False)
        path = models / f'rgb/{seed}/checkpoint.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: state[k] for k in ['head', 'adapters']}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=HERE / 'assets/features')
    parser.add_argument('--branch', choices=['trees', 'rgb', 'sam', 'all'], default='all')
    parser.add_argument('--work', type=Path, default=HERE.parent / 'output/retrain')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--hours', type=float, default=2, help='Soft wall-time budget; saved stages can be resumed')
    parser.add_argument('--inspect', action='store_true', help='Validate training inputs without fitting or requiring CUDA')
    args = parser.parse_args(); public, sam = validate(args.features)
    if args.inspect:
        print(json.dumps(dict(public=1299, source_groups=len(set(public['sources'])),
                              SAM_eligible=int(((sam['mask'].sum(1) >= 3) & sam['mask'][:, -1]).sum()),
                              trees=24, RGB_seeds=3, SAM_seeds=3, epochs_RGB=12, epochs_SAM=36,
                              no_private_training=True), indent=2))
        return
    assert torch.cuda.is_available(), 'CUDA GPU required; no automatic CPU training fallback'
    torch.set_num_threads(4)
    args.work.mkdir(parents=True, exist_ok=True); models = args.work / 'models'; models.mkdir(exist_ok=True)
    protocol = dict(features={p.name: sha(p) for p in args.features.glob('*') if p.is_file()},
                    pretrained=sha(args.pretrained / 'model.ckpt') if args.pretrained else None,
                    seeds=list(SEEDS), train=1299, excluded=['0526_public.mp4'],
                    tree_iterations=450, RGB_epochs=12, SAM_epochs=36, T=1.,
                    validation_selection=False, private_labels=False)
    path = args.work / 'training_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Changed inputs: choose a new work directory'
    else:
        dump(path, protocol)
    deadline = time.monotonic() + args.hours*3600
    if args.branch in ['trees', 'all']:
        train_trees(public, models, deadline)
    if args.branch in ['rgb', 'all']:
        train_rgb(args, models, args.work, deadline)
    if args.branch in ['sam', 'all']:
        train_sam(args.features, models, args.work, deadline)
    print('Requested training stages completed. Regenerate private RGB features with these adapters before inference.')


if __name__ == '__main__':
    main()
