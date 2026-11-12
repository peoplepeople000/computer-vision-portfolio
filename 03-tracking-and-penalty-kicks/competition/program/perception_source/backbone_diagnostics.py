"""Bounded follow-up diagnostics on immutable backbone-first feature caches."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import rgb_gru as model
import rgb_gru_data as data


def class_weights(training_rows):
    """A mild inverse-frequency correction; fitted to training labels only."""
    result = {}
    for key, labels in data.TARGETS.items():
        counts = training_rows[key].value_counts().reindex(labels, fill_value=0).to_numpy()
        if np.any(counts == 0):
            raise ValueError(f'Missing training class: {key}')
        weights = np.sqrt(counts.sum() / counts)
        result[key] = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    return result


def select_memory_rows(training_rows, count=24, seed=6643):
    if (training_rows.fold == 0).any():
        raise ValueError('Memory diagnostic must not use validation clips')
    shuffled = training_rows.sample(frac=1, random_state=seed)
    chosen, sources = [], set()
    for key, labels in data.TARGETS.items():
        for label in labels:
            candidates = shuffled[(shuffled[key] == label) & ~shuffled.source_id.isin(sources)]
            if candidates.empty:
                raise ValueError('Need independent training sources for memory diagnostic')
            row = candidates.iloc[0]
            chosen.append(row['clip']); sources.add(row.source_id)
    for _, row in shuffled.iterrows():
        if len(chosen) >= count:
            break
        if row.source_id not in sources:
            chosen.append(row['clip']); sources.add(row.source_id)
    if len(chosen) != count:
        raise ValueError('Insufficient training sources')
    return training_rows.set_index('clip').loc[chosen].reset_index()


def perturb_batch(batch, mode):
    """Inference intervention, not a separately trained metadata-only baseline."""
    result = dict(batch)
    result['inputs'] = batch['inputs'].clone()
    if mode == 'zero_visual':
        result['inputs'].zero_()
    elif mode == 'reverse_visual':
        for i, n in enumerate(batch['lengths'].tolist()):
            result['inputs'][i, :n] = batch['inputs'][i, :n].flip(0)
        # Keep timestamps fixed: reverse the visual sequence at original times.
    elif mode != 'normal':
        raise ValueError(mode)
    return result


def audit(previous, out, public_csv=None):
    cache = previous / 'data/processed/backbone_scene_v1'
    rows = pd.read_csv(cache / 'public_folds.csv')
    assert len(rows) == 1299 and not rows['clip'].duplicated().any()
    assert rows.groupby('source_id').fold.nunique().max() == 1
    assert set(data.CONFIG['excluded']).isdisjoint(rows['clip'])
    protocol = json.loads((cache / 'split_protocol.json').read_text())
    assert data.sha(cache / 'public_folds.csv') == protocol['fold_csv_sha256']
    if public_csv is not None:
        assert data.sha(public_csv) == protocol['public_csv_sha256']
        original = pd.read_csv(public_csv).set_index('clip').loc[rows['clip']].reset_index()
        pd.testing.assert_frame_equal(original[rows.columns.drop('fold')], rows.drop(columns='fold'))
    model.validate_caches(rows, cache, 'features')
    counts, tail_counts, movement, norms, hashes = [], [], [], [], set()
    for clip in rows['clip']:
        with np.load(cache / 'features' / (clip + '.npz'), allow_pickle=False) as a:
            x, mask, ts, frame = a['features'], a['view_mask'], a['timestamps'], a['frame_indices']
            assert np.isfinite(x).all() and np.isfinite(ts).all(), clip
            assert mask[:, 0].all() and not mask[:, 1:].any(), clip
            assert np.all(np.diff(ts) > 0) and np.all(np.diff(frame) > 0), clip
            assert x.shape == (len(ts), 3, 1280), clip
            counts.append(len(ts)); tail_counts.append(int((ts >= ts[-1] - .6).sum()))
            norms.append(float(np.linalg.norm(x[:, 0], axis=1).mean()))
            movement.append(float(np.abs(np.diff(x[:, 0], axis=0)).mean()))
            hashes.add(hashlib.sha256(x[:, 0].tobytes()).hexdigest())
    describe = lambda x: dict(zip(['min', 'median', 'max'], map(float, np.quantile(x, [0, .5, 1]))))
    result = {'clips': len(rows), 'raw_labels_verified': public_csv is not None,
              'fold_sha256': data.sha(cache / 'public_folds.csv'),
              'class_counts': {part: {key: group[key].value_counts().to_dict() for key in data.TARGETS}
                               for part, group in [('train', rows[rows.fold != 0]), ('validation', rows[rows.fold == 0])]},
              'sampled_frames': describe(counts), 'tail_frames': describe(tail_counts),
              'feature_norm': describe(norms), 'adjacent_feature_change': describe(movement),
              'distinct_feature_sequences': len(hashes), 'finite_and_chronological': True}
    data.dump(out / 'input_audit.json', result)
    return rows, cache


class BudgetStop(RuntimeError):
    pass


def batches(items, seed, shuffle=False, batch_size=32):
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=model.collate,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def memory_test(items, selected, out, deadline, provenance):
    """Training-set-only learnability check; never a validation result."""
    model.seed_all(6643)
    net = model.Predictor(config={**model.MODEL_CONFIG, 'dropout': 0.}).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=0.)
    generator = torch.Generator().manual_seed(6643)
    batch = model.to_device(model.collate([items[c] for c in selected['clip']]), 'cuda')
    config = {'purpose': 'training-only memory diagnostic', 'clips': selected['clip'].tolist(),
              'seed': 6643, 'max_steps': 300, 'lr': .003, 'dropout': 0., **provenance}
    data.immutable_json(out / 'memory_config.json', config)
    checkpoint = out / 'memory_last.pt'
    start = 0
    if checkpoint.exists():
        saved = model.restore_training(checkpoint, net, optimizer, generator, config)
        start = saved['step']
        if saved['result']['passed']:
            return saved['result']
    for step in range(start, 300):
        net.train(); optimizer.zero_grad(set_to_none=True)
        logits, _ = net(batch); loss = model.loss_for(logits, batch['labels'])
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); optimizer.step()
        if (step + 1) % 25 == 0 or time.monotonic() > deadline:
            with torch.inference_mode():
                net.eval(); logits, _ = net(batch)
                predictions = torch.stack([logits[k].argmax(1) for k in data.TARGETS], 1)
                metrics = model.metric_report(batch['labels'].cpu(), predictions.cpu())
            passed = all(metrics[k] >= .95 for k in data.TARGETS)
            result = {'steps': step + 1, 'training_only': True, 'passed': passed,
                      'metrics': metrics, 'loss': float(loss.detach()), 'device': str(predictions.device)}
            model.atomic_save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                               'rng': model.capture_rng(generator), 'config': config,
                               'step': step + 1, 'result': result}, checkpoint)
            data.dump(out / 'memory_test.json', result)
            print(json.dumps({'stage': 'memory_test', **result}), flush=True)
            if passed:
                return result
            if time.monotonic() > deadline:
                raise BudgetStop('Memory checkpoint saved')
    return json.loads((out / 'memory_test.json').read_text())


def train_trial(name, seed, reweight, train_items, val_items, rows, out, deadline, provenance):
    run = out / name
    weights = class_weights(rows[rows.fold != 0]) if reweight else None
    config = {'model': {**model.MODEL_CONFIG, 'aggregation': 'gru'}, 'seed': seed,
              'epochs': 8, 'batch_size': 32, 'lr': .001, 'weight_decay': .01,
              'class_weights': {k: v.tolist() for k, v in weights.items()} if weights else None,
              'train_clips': rows.loc[rows.fold != 0, 'clip'].tolist(),
              'validation_clips': rows.loc[rows.fold == 0, 'clip'].tolist(), **provenance}
    data.immutable_json(run / 'config.json', config)
    model.seed_all(seed)
    net = model.Predictor(config=config['model']).cuda()
    initial_hash = tensor_digest(net.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    train_batches = batches(train_items, seed, True)
    validation_batches = batches(val_items, seed)
    train_evaluation = batches(train_items, seed)
    history, best, start = [], -1., 0
    if (run / 'last.pt').exists():
        saved = model.restore_training(run / 'last.pt', net, optimizer, train_batches.generator, config)
        history, best, start = saved['history'], saved['best_score'], saved['epoch']
    if weights:
        weights = {k: v.cuda() for k, v in weights.items()}
    for epoch in range(start, 8):
        net.train(); losses = []
        for batch in train_batches:
            batch = model.to_device(batch, 'cuda'); optimizer.zero_grad(set_to_none=True)
            logits, _ = net(batch)
            loss = (sum(w * torch.nn.functional.cross_entropy(logits[k], batch['labels'][:, i], weight=weights[k])
                        for i, (k, w) in enumerate(zip(data.TARGETS, data.WEIGHTS)))
                    if weights else model.loss_for(logits, batch['labels']))
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
            optimizer.step(); losses.append(float(loss.detach()))
        metrics, records = model.evaluate(net, validation_batches, 'cuda')
        train_metrics, _ = model.evaluate(net, train_evaluation, 'cuda')
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                        'train': train_metrics, 'validation': metrics})
        if metrics['weighted_score'] > best:
            best = metrics['weighted_score']
            model.atomic_save({'state_dict': net.state_dict(), 'config': config, 'epoch': epoch + 1,
                               'metrics': metrics, 'initial_state_sha256': initial_hash}, run / 'best.pt')
        model.atomic_save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                           'config': config, 'epoch': epoch + 1, 'history': history, 'best_score': best,
                           'rng': model.capture_rng(train_batches.generator)}, run / 'last.pt')
        data.dump(run / 'history.json', history)
        print(json.dumps({'stage': name, 'epoch': epoch + 1, 'train_score': train_metrics['weighted_score'],
                          'validation_score': metrics['weighted_score']}), flush=True)
        if time.monotonic() > deadline and epoch + 1 < 8:
            raise BudgetStop(name + ': epoch checkpoint saved')
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(saved['state_dict'])
    metrics, records = model.evaluate(net, validation_batches, 'cuda')
    assert abs(metrics['weighted_score'] - saved['metrics']['weighted_score']) < 1e-12
    interventions = {}
    for mode in ['reverse_visual', 'zero_visual']:
        altered = (perturb_batch(b, mode) for b in batches(val_items, seed))
        intervention_metrics, intervention_records = model.evaluate(net, altered, 'cuda')
        interventions[mode] = intervention_metrics
        data.dump(run / (mode + '_predictions.json'), intervention_records)
    data.dump(run / 'validation_predictions.json', records)
    summary = {'best_epoch': saved['epoch'], 'metrics': metrics, 'interventions': interventions,
               'initial_state_sha256': initial_hash, 'checkpoint_sha256': data.sha(run / 'best.pt'),
               'note': 'Validation-selected epoch; interventions are distribution shifts, not trained controls.'}
    data.dump(run / 'summary.json', summary)
    return summary


def execute(previous, out, public_csv, max_seconds):
    started = time.monotonic(); deadline = started + max_seconds - 120
    assert torch.cuda.is_available(), 'CUDA required; no CPU training fallback'
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'input_manifest_sha256': data.sha(previous / 'artifact_manifest.json'),
                  'code_sha256': {Path(p).name: data.sha(p) for p in [__file__, model.__file__, data.__file__]}}
    try:
        rows, cache = audit(previous, out, public_csv)
        provenance['fold_sha256'] = data.sha(cache / 'public_folds.csv')
        encoder, manifest = model.make_encoder(previous / 'models/efficientnet-b0')
        encoder.eval().cuda()
        with torch.inference_mode(): probe = encoder(torch.zeros(1, 3, 224, 224, device='cuda'))
        assert probe.is_cuda and probe.shape == (1, 1280)
        data.dump(out / 'gpu_check.json', {'gpu': torch.cuda.get_device_name(0),
                  'device': str(probe.device), 'torch': str(torch.__version__), 'encoder': manifest})
        del encoder, probe
        dataset = model.ClipDataset(rows, cache, 'features', 'scene')
        items = {item['clip']: item for item in dataset}
        selected = select_memory_rows(rows[rows.fold != 0])
        memory = memory_test(items, selected, out, deadline, provenance)
        if not memory['passed']:
            data.dump(out / 'status.json', {'stage': 'diagnostic_failed', 'reason': 'Memory test failed; no full trials started'})
            return
        train_items = [items[c] for c in rows.loc[rows.fold != 0, 'clip']]
        val_items = [items[c] for c in rows.loc[rows.fold == 0, 'clip']]
        results = {}
        for seed in [6643, 6644, 6645]:
            for reweight in [False, True]:
                name = ('sqrt_weighted' if reweight else 'control') + f'_{seed}'
                if time.monotonic() > deadline:
                    raise BudgetStop('Ready for ' + name)
                results[name] = train_trial(name, seed, reweight, train_items, val_items, rows,
                                            out, deadline, provenance)
            assert results[f'control_{seed}']['initial_state_sha256'] == results[f'sqrt_weighted_{seed}']['initial_state_sha256']
        data.dump(out / 'comparison.json', results)
        data.dump(out / 'status.json', {'stage': 'complete', 'experiments': list(results),
                  'elapsed_seconds': time.monotonic() - started, 'private_submission': False})
    except BudgetStop as exc:
        data.dump(out / 'status.json', {'stage': 'checkpointed', 'reason': str(exc)})
    except Exception as exc:
        data.dump(out / 'status.json', {'stage': 'error', 'error_type': type(exc).__name__, 'reason': str(exc)})
        raise
    finally:
        artifacts = {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*')
                     if p.is_file() and p.name != 'artifact_manifest.json' and p.suffix != '.tmp'}
        data.dump(out / 'artifact_manifest.json', artifacts)
        print((out / 'status.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'run'])
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public-csv', type=Path)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.stage == 'run':
        execute(args.previous, args.out, args.public_csv, args.max_seconds)
    else:
        audit(args.previous, args.out, args.public_csv)
        print((args.out / 'input_audit.json').read_text())
