import math, time
from pathlib import Path
import numpy as np
import torch
from sam_core import TokenHead, read, dump, sha, digest
from native_body import lr_value, snapshot, save
SEEDS = (6643, 6644, 6645)

def temperature(logits, labels):
    z = np.asarray(logits, float)
    y = np.asarray(labels, int)
    assert z.ndim == 3 and z.shape[0] == 3 and (z.shape[1] == len(y)) and (z.shape[2] == 3) and np.isin(y, [0, 1, 2]).all()

    def nll(logt):
        v = z / np.exp(logt)
        m = v.max(-1)
        lse = m + np.log(np.exp(v - m[..., None]).sum(-1))
        return float((lse - np.take_along_axis(v, np.broadcast_to(y[None, :, None], (3, len(y), 1)), axis=-1)[..., 0]).mean())
    lo, hi = (np.log(0.5), np.log(5.0))
    ratio = (math.sqrt(5) - 1) / 2
    c = hi - ratio * (hi - lo)
    d = lo + ratio * (hi - lo)
    for _ in range(100):
        if nll(c) < nll(d):
            hi = d
            d = c
            c = hi - ratio * (hi - lo)
        else:
            lo = c
            c = d
            d = lo + ratio * (hi - lo)
    best = min([0.0, np.log(0.5), np.log(5.0), (lo + hi) / 2], key=nll)
    return dict(T=float(np.exp(best)), before=nll(0.0), after=nll(best), clips=len(y), seed_instances=3 * len(y), method='train-side NLL,golden-section,one common scalar')

def fit(root, out, task, epochs, data, rows, labels, protocol_sha, deadline):
    allowed = {'outer1': {2, 3, 4}, 'outer2': {3, 4}, 'inner4': {3}, 'full': {0, 1, 2, 3, 4}}[task]
    folds = np.array([r['fold'] for r in rows])
    sources = np.array([r['source'] for r in rows])
    m = data[1].cpu().numpy()
    eligible = (m.sum(1) >= 3) & m[:, -1]
    fit = np.flatnonzero(np.isin(folds, list(allowed)) & eligible)
    assert (labels[fit] >= 0).all() and (labels[~np.isin(folds, list(allowed))] == -1).all()
    weights = np.array([1 / np.sqrt(sum(sources[fit] == sources[i])) for i in fit], np.float32)
    weights /= weights.mean()
    nb = math.ceil(len(fit) / 16)
    dump(out / task / 'audit.json', dict(fit=fit.tolist(), allowed_folds=sorted(allowed), source_ids=sorted(set(sources[fit])), weights=weights.tolist(), eligible=eligible.tolist(), epochs=epochs))
    for seed in SEEDS:
        f = out / task / str(seed)
        f.mkdir(parents=True, exist_ok=True)
        cp = f / 'checkpoint.pt'
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        head = TokenHead().cuda()
        initial = digest(head.state_dict())
        optimizer = None
        seen = np.zeros(len(rows), int)
        history = []
        updates = 0
        epoch0 = 0
        if cp.exists():
            ck = torch.load(cp, map_location='cuda', weights_only=False)
            assert ck['protocol_sha256'] == protocol_sha and ck['initial_hash'] == initial
            head.load_state_dict(ck['head'])
            seen = ck['seen']
            history = ck['history']
            updates = ck['updates']
            epoch0 = ck['next_epoch']
            torch.set_rng_state(ck['rng_cpu'].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
        for epoch in range(epoch0, epochs):
            assert time.monotonic() < deadline, 'Safe epoch budget boundary'
            if optimizer is None or epoch == 12:
                optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.02)
                if cp.exists() and epoch == epoch0 and (epoch not in [0, 12]):
                    optimizer.load_state_dict(ck['optimizer'])
            start = 0 if epoch < 12 else 12
            length = 12 if epoch < 12 else 24
            order = np.random.default_rng(seed * 100 + epoch).permutation(len(fit))
            head.train()
            loss_sum = 0.0
            for bi, off in enumerate(range(0, len(order), 16)):
                ix = order[off:off + 16]
                ids = fit[ix]
                for g in optimizer.param_groups:
                    g['lr'] = 0.0003 * lr_value((epoch - start) * nb + bi, length * nb)
                optimizer.zero_grad(set_to_none=True)
                logits = head(*[x[ids] for x in data], 'B_sequence')
                loss = (torch.nn.functional.cross_entropy(logits, torch.tensor(labels[ids], device='cuda'), label_smoothing=0.05, reduction='none') * torch.tensor(weights[ix], device='cuda')).mean()
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                assert torch.isfinite(grad)
                optimizer.step()
                updates += 1
                seen[ids] += 1
                loss_sum += float(loss) * len(ids)
            history.append(dict(epoch=epoch + 1, loss=loss_sum / len(fit), updates=updates))
            state = dict(head=head.state_dict(), optimizer=optimizer.state_dict(), next_epoch=epoch + 1, updates=updates, seen=seen, history=history, initial_hash=initial, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(), protocol_sha256=protocol_sha, task=task, seed=seed)
            snapshot(cp, state)
            if epoch + 1 in [12, 36]:
                snapshot(f / f'epoch{epoch + 1}.pt', state)
                head.eval()
                values = []
                with torch.inference_mode():
                    for off in range(0, len(rows), 64):
                        values.append(head(*[x[off:off + 64] for x in data], 'B_sequence').cpu().numpy())
                values = np.concatenate(values)
                save(f / f'epoch{epoch + 1}.npz', logits=values)
                restored = TokenHead().cuda().eval()
                restored.load_state_dict(torch.load(f / f'epoch{epoch + 1}.pt', map_location='cuda', weights_only=False)['head'])
                with torch.inference_mode():
                    replay = np.concatenate([restored(*[x[o:o + 64] for x in data], 'B_sequence').cpu().numpy() for o in range(0, len(rows), 64)])
                np.testing.assert_allclose(values, replay, atol=1e-05, rtol=1e-05)
                assert digest(state['head']) != initial
                dump(f / f'epoch{epoch + 1}_reload.json', dict(actual_cuda=True, max_error=float(abs(values - replay).max()), fit=len(fit), updates=updates, checkpoint_sha256=sha(f / f'epoch{epoch + 1}.pt')))
                del restored
            print(__import__('json').dumps(dict(stage='epoch', task=task, seed=seed, epoch=epoch + 1, updates=updates)), flush=True)
        np.testing.assert_array_equal(seen[fit], np.full(len(fit), epochs))
        assert not seen[np.setdiff1d(np.arange(len(rows)), fit)].any()
        print(__import__('json').dumps(dict(stage='fit_complete', task=task, seed=seed, epochs=epochs)), flush=True)
        del head, optimizer
        torch.cuda.empty_cache()
