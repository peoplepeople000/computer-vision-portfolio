"""R1: bounded GPU linear probes on frozen sparse RGB features.

Select regularization on inner source fold 1; evaluate outer fold 0 once.
No detections, encoder inference, private data, or outer-label model selection.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rgb_gru as model
import rgb_gru_data as data
import sparse_role_comparison as sparse

VARIANTS = ('foot_linear', 'last_scene', 'last_roles', 'mean_roles', 'delta_roles')
LAMBDAS = (0.0001, 0.001, 0.01)
STEPS = 80


def representation(items, variant):
    """Missing role values cannot enter means/differences; include validity flags."""
    output = []
    for item in items:
        x, mask = item['inputs'], item['view_mask']
        clean = x.masked_fill(~mask[..., None], 0)
        if variant == 'foot_linear':
            value = torch.empty(0)
        elif variant == 'last_scene':
            value = clean[-1, 0]
        elif variant == 'last_roles':
            value = torch.cat([clean[-1].flatten(), mask[-1].float()])
        elif variant in ('mean_roles', 'delta_roles'):
            mean = clean.sum(0) / mask.sum(0).clamp_min(1)[:, None]
            value = torch.cat([mean.flatten(), mask.float().mean(0)])
            if variant == 'delta_roles':
                # Ordered adjacent differences, not physical velocities or tracks.
                diff = clean.new_zeros((2, 3, x.shape[-1]))
                valid = torch.zeros((2, 3), dtype=torch.bool)
                n = min(len(x) - 1, 2)
                if n:
                    valid[:n] = mask[1:n + 1] & mask[:n]
                    diff[:n] = (clean[1:n + 1] - clean[:n]).masked_fill(~valid[:n, :, None], 0)
                value = torch.cat([value, diff.flatten(), valid.flatten().float()])
        else:
            raise ValueError(variant)
        output.append(value)
    return torch.stack(output)


def standardize(x, fit_indices):
    """Fit every transformation on this fit's training rows only."""
    mean = x[fit_indices].mean(0)
    scale = x[fit_indices].std(0, unbiased=False).clamp_min(0.05)
    transformed = ((x - mean) / scale).clamp(-10, 10) / math.sqrt(max(1, x.shape[1]))
    return transformed, mean, scale


def foot_priors(metadata, labels, fit_indices):
    foot = metadata.argmax(1)
    priors = []
    for k, classes in enumerate(data.TARGETS.values()):
        global_count = torch.bincount(labels[fit_indices, k], minlength=len(classes)).float() + 1
        rows = []
        for f in range(3):
            selected = fit_indices[foot[fit_indices] == f]
            counts = (torch.bincount(labels[selected, k], minlength=len(classes)).float() + 1
                      if len(selected) else global_count)
            rows.append((counts / counts.sum()).log())
        priors.append(torch.stack(rows))
    return torch.cat(priors, 1)[foot]


def predict_metrics(logits, labels, indices):
    probabilities = [p.softmax(1) for p in logits.split((3, 3, 2), 1)]
    predictions = torch.stack([p.argmax(1) for p in probabilities], 1)
    metrics = model.metric_report(labels[indices].cpu().numpy(), predictions[indices].cpu().numpy())
    return metrics, predictions, probabilities


def fit(x, metadata, labels, fit_indices, evaluation, strength, out, deadline,
        residual=True):
    started = time.monotonic()
    z, mean, scale = standardize(x, fit_indices)
    features = torch.cat([z, metadata], 1).cuda()
    offsets = foot_priors(metadata, labels, fit_indices).cuda() if residual else features.new_zeros((len(x), 8))
    y = labels.cuda(); fit_gpu = fit_indices.cuda()
    head = nn.Linear(features.shape[1], 8).cuda()
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1, max_iter=1, history_size=10,
                                 line_search_fn='strong_wolfe')
    history = []
    for step in range(STEPS):
        def closure():
            optimizer.zero_grad()
            logits = head(features[fit_gpu]) + offsets[fit_gpu]
            losses = [nn.functional.cross_entropy(a, y[fit_gpu, k])
                      for k, a in enumerate(logits.split((3, 3, 2), 1))]
            loss = sum(w * v for w, v in zip(data.WEIGHTS, losses)) + strength * head.weight.square().sum() / 2
            assert torch.isfinite(loss)
            loss.backward()
            return loss
        objective = float(optimizer.step(closure).detach())
        if step == 0 or (step + 1) % 10 == 0:
            with torch.no_grad():
                logits = head(features) + offsets
                train_metrics, _, _ = predict_metrics(logits, y, fit_gpu)
                entry = {'step': step + 1, 'objective': objective,
                         'train_score': train_metrics['weighted_score'],
                         'elapsed_seconds': time.monotonic() - started}
                if evaluation is not None:
                    entry['inner_score'] = predict_metrics(logits, y, evaluation.cuda())[0]['weighted_score']
                history.append(entry)
                data.dump(out / 'history.json', history)
        if time.monotonic() > deadline:
            model.atomic_save({'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
                               'step': step + 1, 'history': history}, out / 'interrupted.pt')
            raise TimeoutError('Budget ended; partial checkpoint saved; no automatic restart')
    with torch.no_grad():
        logits = (head(features) + offsets).cpu()
    torch.cuda.synchronize()
    checkpoint = {'model': {k: v.cpu() for k, v in head.state_dict().items()},
                  'optimizer': optimizer.state_dict(), 'step': STEPS, 'mean': mean, 'scale': scale,
                  'fit_indices': fit_indices, 'regularization': strength, 'residual': residual,
                  'prior_tables_note': 'Offsets recomputed only from fit_indices labels and metadata',
                  'offsets': offsets.cpu(), 'history': history, 'seed': 6643}
    model.atomic_save(checkpoint, out / 'checkpoint.pt')
    saved = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True)
    check = nn.Linear(features.shape[1], 8); check.load_state_dict(saved['model'])
    with torch.no_grad():
        restored = check(features.cpu()) + saved['offsets']
    torch.testing.assert_close(restored, logits, atol=2e-5, rtol=2e-5)
    data.dump(out / 'history.json', history)
    print(json.dumps({'stage': 'fit_complete', 'fit': str(out),
                      'seconds': time.monotonic() - started,
                      'parameters': sum(p.numel() for p in head.parameters())}), flush=True)
    return logits, {'seconds': time.monotonic() - started,
                    'parameters': sum(p.numel() for p in head.parameters()),
                    'checkpoint_sha256': data.sha(out / 'checkpoint.pt')}


def execute(cache, out, budget=600):
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4); model.seed_all(6643)
    started = time.monotonic(); deadline = started + budget - 60
    rows, _, _, lookup, provenance = sparse.load_bundle(cache, cache / 'public_folds.csv')
    items = [lookup[c] for c in rows['clip']]
    metadata = torch.stack([i['metadata'] for i in items]); labels = torch.stack([i['labels'] for i in items])
    ix = lambda mask: torch.tensor(np.flatnonzero(mask), dtype=torch.long)
    train, val = ix(rows.fold != 0), ix(rows.fold == 0)
    inner_train, inner_val = ix(~rows.fold.isin([0, 1])), ix(rows.fold == 1)
    assert set(rows.iloc[train].source_id).isdisjoint(rows.iloc[val].source_id)
    assert set(rows.iloc[inner_train].source_id).isdisjoint(rows.iloc[inner_val].source_id)
    protocol = {**provenance, 'source_sha256': data.sha(__file__), 'variants': list(VARIANTS),
                'lambdas': list(LAMBDAS), 'steps': STEPS, 'outer_fold': 0, 'inner_fold': 1,
                'train_clips': rows.iloc[train]['clip'].tolist(), 'validation_clips': rows.iloc[val]['clip'].tolist(),
                'inner_train_clips': rows.iloc[inner_train]['clip'].tolist(),
                'inner_validation_clips': rows.iloc[inner_val]['clip'].tolist(),
                'note': 'No outer-fold epoch selection. Differences are appearance changes, not tracked pose or physical motion.'}
    data.immutable_json(out / 'protocol.json', protocol)
    data.dump(out / 'runtime.json', {'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__)})
    baseline_logits = foot_priors(metadata, labels, train)
    baseline, base_predictions, _ = predict_metrics(baseline_logits, labels, val)
    results = {'foot_rule': {'metrics': baseline}}
    try:
        for variant in VARIANTS:
            directory = out / variant
            if (directory / 'result.json').exists():
                results[variant] = json.loads((directory / 'result.json').read_text()); continue
            x = representation(items, variant); candidates = []
            for strength in LAMBDAS:
                logits, cost = fit(x, metadata, labels, inner_train, inner_val, strength,
                                   directory / f'inner_{strength}', deadline, residual=variant != 'foot_linear')
                score = predict_metrics(logits, labels, inner_val)[0]['weighted_score']
                candidates.append({'strength': strength, 'inner_score': score, **cost})
            chosen = max(candidates, key=lambda a: (a['inner_score'], a['strength']))['strength']
            logits, cost = fit(x, metadata, labels, train, None, chosen, directory / 'outer', deadline,
                               residual=variant != 'foot_linear')
            metrics, predictions, probabilities = predict_metrics(logits, labels, val)
            delta = ((predictions[val] == labels[val]).float() -
                     (base_predictions[val] == labels[val]).float()) @ torch.tensor(data.WEIGHTS)
            result = {'metrics': metrics, 'chosen_strength': chosen, 'candidates': candidates, **cost,
                      'vs_foot': sparse.source_interval(delta.numpy(), rows.iloc[val].source_id)}
            records = [{'clip': rows.iloc[int(i)]['clip'], 'truth': labels[i].tolist(),
                        'prediction': predictions[i].tolist(),
                        'probabilities': [p[i].tolist() for p in probabilities]} for i in val]
            data.dump(directory / 'predictions.json', records); data.dump(directory / 'result.json', result)
            results[variant] = result
            print(json.dumps({'stage': variant, 'score': metrics['weighted_score'],
                              'fit_seconds': cost['seconds'], 'elapsed_seconds': time.monotonic() - started}), flush=True)
        data.dump(out / 'summary.json', results)
        data.dump(out / 'status.json', {'stage': 'complete', 'seconds': time.monotonic() - started})
    except Exception as exc:
        data.dump(out / 'status.json', {'stage': 'error', 'error': str(exc)}); raise
    finally:
        data.dump(out / 'artifact_manifest.json', {str(p.relative_to(out)): data.sha(p) for p in out.rglob('*')
                  if p.is_file() and p.name != 'artifact_manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=600)
    args = parser.parse_args(); execute(args.cache, args.out, args.budget)
