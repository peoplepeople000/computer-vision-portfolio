"""R142 fixed full-public schedule; fresh adapters/head and resumable state."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import human_rgb_adaptation as adapt


def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def atomic(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(path)
def digest(state):
    h = hashlib.sha256()
    for n, v in sorted(state.items()):h.update(n.encode()); h.update(v.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
def schedule(seed, epoch, n=1299):return np.random.default_rng(seed*100+epoch).permutation(n)
def batches(rows):return [rows[i:i+16] for i in range(0, len(rows), 16)]
def reset_adapters(model, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for name, p in adapt.adapter_parameters(model).items():
            if name.endswith(('Aq', 'Av')):torch.nn.init.kaiming_uniform_(p, a=5**.5)
            else:p.zero_()


def fit(model, head_initial, seed, arm, rows, labels, feature, folder, protocol_sha, base_digest, timeout):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); head = adapt.DirectionHead().cuda().float(); head.load_state_dict(head_initial, strict=True)
    adapters = adapt.adapter_parameters(model) if arm == 'adapted' else {}
    initial = dict(head={n:v.clone() for n, v in head_initial.items()}, adapters={n:p.detach().cpu().clone() for n, p in adapters.items()},
                   head_hash=digest(head_initial), adapter_hash=digest(adapters))
    if (folder/'initial.pt').exists():
        old = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)
        assert old['head_hash'] == initial['head_hash'] and old['adapter_hash'] == initial['adapter_hash']
    else:atomic(folder/'initial.pt', initial)
    params = list(head.parameters())+list(adapters.values())
    opt = torch.optim.AdamW(params, lr=.0003, weight_decay=.01); scaler = torch.amp.GradScaler('cuda')
    epoch0 = batch0 = updates = 0; history = []; seen = np.zeros(len(rows), np.int64)
    if (folder/'checkpoint.pt').exists():
        ck = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        assert ck['protocol_sha256'] == protocol_sha and ck['arm'] == arm and ck['seed'] == seed and ck['base_hash'] == base_digest
        head.load_state_dict(ck['head'], strict=True)
        with torch.no_grad():
            for n, p in adapters.items():p.copy_(ck['adapters'][n].to(p))
        opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        epoch0, batch0, updates, history, seen = ck['next_epoch'], ck['next_batch'], ck['updates'], ck['history'], ck['seen']
        torch.set_rng_state(ck['rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    def save(epoch, batch):
        atomic(folder/'checkpoint.pt', dict(protocol_sha256=protocol_sha, arm=arm, seed=seed, base_hash=base_digest, next_epoch=epoch, next_batch=batch,
             updates=updates, seen=seen, history=history, head={n:v.detach().cpu().clone() for n, v in head.state_dict().items()},
             adapters={n:p.detach().cpu().clone() for n, p in adapters.items()}, optimizer=opt.state_dict(), scaler=scaler.state_dict(),
             rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    for epoch in range(epoch0, 12):
        order = schedule(seed, epoch, len(rows)); path = folder/f'schedule_{epoch}.npy'
        if path.exists():np.testing.assert_array_equal(np.load(path), order)
        else:np.save(path, order)
        for bi, batch in enumerate(batches(order)):
            if epoch == epoch0 and bi < batch0:continue
            if timeout():save(epoch, bi); return None
            opt.zero_grad(set_to_none=True); total = 0.
            for j in batch:
                feat = feature(int(j), arm, True)
                logits = head(feat, torch.tensor([rows[j]['available']], device='cuda'))
                loss = F.cross_entropy(logits, torch.tensor([int(labels[j])], device='cuda'))/len(batch)
                scaler.scale(loss).backward(); total += float(loss.detach())
            scaler.unscale_(opt)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapters.values())
            torch.nn.utils.clip_grad_norm_(params, 1.); scaler.step(opt); scaler.update(); updates += 1; seen[batch] += 1
            assert all(int(opt.state[p]['step']) == updates for p in head.parameters())
            history.append(dict(epoch=epoch, batch=bi, positions=batch.tolist(), mean_loss=total, update=updates))
            ne, nb = (epoch+1, 0) if bi+1 == len(batches(order)) else (epoch, bi+1)
            if updates%25 == 0 or nb == 0 or timeout():save(ne, nb)
        print(json.dumps(dict(stage='epoch', arm=arm, seed=seed, epoch=epoch+1, updates=updates)), flush=True)
    assert updates == 12*len(batches(np.arange(len(rows)))); np.testing.assert_array_equal(seen, np.full(len(rows), 12))
    assert digest(head.state_dict()) != initial['head_hash'] and adapt.base_hash(model) == base_digest
    if adapters:
        assert all(not torch.equal(p.detach().cpu(), initial['adapters'][n]) for n, p in adapters.items())
        assert all(int(opt.state[p]['step']) == sum(any(rows[int(j)]['available'] for j in batch) for epoch in range(12) for batch in batches(schedule(seed, epoch, len(rows)))) for p in adapters.values())
    save(12, 0)
    dump(folder/'training_audit.json', dict(arm=arm, seed=seed, epochs=12, updates=updates, head_initial_hash=initial['head_hash'],
        adapter_initial_hash=initial['adapter_hash'], base_after=adapt.base_hash(model), base_before=base_digest,
        adapter_deltas={n:float((p.detach().cpu()-initial['adapters'][n]).abs().max()) for n, p in adapters.items()},
        seen=seen.tolist(), head_final_hash=digest(head.state_dict())))
    return head
