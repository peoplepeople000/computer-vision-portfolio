"""Frozen R95 image-representation probe; no perception or neural-head training."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def select(rows, count):
    """Deterministic source-balanced selection within each canonical class."""
    result = []
    for label in range(3):
        pool = [r for r in rows if r['canonical_label'] == label]
        used = {}
        for _ in range(count):
            assert pool, (label, count)
            chosen = min(pool, key=lambda r: (used.get(r['source'], 0),
                hashlib.sha256(('r95|' + r['clip']).encode()).hexdigest()))
            result.append(chosen)
            used[chosen['source']] = used.get(chosen['source'], 0) + 1
            pool.remove(chosen)
    return result


def ridge(fit, probe, labels):
    """Fixed readout with strictly fit-only normalization and intercept."""
    fit, probe = np.asarray(fit, np.float64), np.asarray(probe, np.float64)
    mean, scale = fit.mean(0), np.maximum(fit.std(0), 1e-6)
    def transform(x):
        x = (x - mean) / scale
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    train, check = transform(fit), transform(probe)
    center = train.mean(0)
    train, check = train - center, check - center
    target = np.eye(3)[labels]
    prior = target.mean(0)
    alpha = np.linalg.solve(train @ train.T + .1 * np.eye(len(train)), target - prior)
    scores = check @ train.T @ alpha + prior
    probabilities = np.exp(scores - scores.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    return probabilities, dict(mean=mean, scale=scale, center=center,
                               train=train, alpha=alpha, prior=prior)


def encoder(checkpoint, kind, revision):
    import torch
    from torchvision.models import efficientnet_b0
    if kind == 'imagenet_b0':
        model = efficientnet_b0(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        return model.features.cuda().eval().requires_grad_(False), {'strict': True}
    model = torch.hub.load('facebookresearch/dinov3:' + revision, 'dinov3_vith16plus',
                          pretrained=False, drop_path=0.0, trust_repo=True)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = blob.get('state_dict', blob)
    prefix = 'backbone.encoder.'
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing = sorted(set(model.state_dict()) - set(state))
    extra = sorted(set(state) - set(model.state_dict()))
    assert missing == ['mask_token'] and not extra, (missing, extra)
    assert torch.equal(model.mask_token, torch.zeros_like(model.mask_token))
    model.load_state_dict(state, strict=False)
    # Every non-default encoder tensor must be exactly the checkpoint value.
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    return model.cuda().eval().requires_grad_(False), {
        'missing': missing, 'unexpected': extra, 'zero_mask_verified': True,
        'loaded_tensors': len(state), 'all_loaded_tensors_equal_checkpoint': True}


def run(inputs, out, checkpoint):
    import gc
    import torch
    from PIL import Image
    import torch.nn.functional as F
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    inputs, out, checkpoint = map(Path, (inputs, out, checkpoint))
    out.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((inputs / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(inputs / name) == expected, name
    plan = manifest['plan']
    assert sha(checkpoint) == plan['sam_checkpoint_sha256']
    assert sha(inputs / 'imagenet.pth') == plan['imagenet_checkpoint_sha256']
    rows = json.loads((inputs / 'images.json').read_text())
    assert len(rows) == 216 and all('label' not in r and 'canonical_label' not in r for r in rows)
    tensors = []
    for row in rows:
        a = np.array(Image.open(inputs / row['file']).convert('RGB'))
        assert a.shape == (384, 384, 3)
        if row['flip']: a = a[:, ::-1].copy()
        x = torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255
        tensors.append((x - torch.tensor([.485,.456,.406])[:,None,None]) /
                       torch.tensor([.229,.224,.225])[:,None,None])
    dump(out / 'protocol.json', dict(manifest=manifest, worker_sha256=sha(__file__),
                                    sam_checkpoint_sha256=sha(checkpoint), labels_uploaded=False))
    start = time.monotonic()
    audits = {}
    def extract(model, kind, x):
        with torch.inference_mode():
            x = x[None].cuda()
            y = model(x) if kind == 'imagenet_b0' else model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]
            assert y.is_cuda and torch.isfinite(y).all()
            y = F.adaptive_avg_pool2d(y, (2,2)).flatten(1)
            return y[0].cpu().numpy()
    for kind, weight in [('imagenet_b0', inputs/'imagenet.pth'), ('sam_body_dinov3', checkpoint)]:
        tick = time.monotonic()
        model, audit = encoder(weight, kind, plan['dinov3_revision'])
        torch.cuda.reset_peak_memory_stats()
        vectors = []
        for i, x in enumerate(tensors):
            assert time.monotonic() - start < 1200, 'Cost limit'
            vectors.append(extract(model, kind, x))
            if (i+1) % 36 == 0: print(json.dumps({'stage':kind, 'completed':i+1, 'total':len(rows)}), flush=True)
        vectors = np.stack(vectors)
        np.save(out / (kind + '.npy'), vectors)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        assert peak < 14
        del model
        gc.collect(); torch.cuda.empty_cache()
        model, reload_audit = encoder(weight, kind, plan['dinov3_revision'])
        replay = np.stack([extract(model, kind, x) for x in tensors[:2]])
        np.testing.assert_allclose(replay, vectors[:2], atol=1e-4, rtol=1e-4)
        audits[kind] = dict(load=audit, reload=reload_audit, features=list(vectors.shape),
                           max_reload_difference=float(np.abs(replay-vectors[:2]).max()),
                           peak_gib=peak, seconds=time.monotonic()-tick, frozen=True)
        dump(out / 'encoder_audit.json', audits)
        del model
        gc.collect(); torch.cuda.empty_cache()
    dump(out / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                                    cuda=torch.version.cuda, seconds=time.monotonic()-start))
    dump(out / 'summary.json', dict(state='complete', images=len(rows), audits=audits,
                                    new_perception_calls=0, new_neural_fits=0))
