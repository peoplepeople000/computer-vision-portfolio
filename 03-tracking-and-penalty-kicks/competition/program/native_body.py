import contextlib
import gc
import io
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from sam_core import ARMS, TokenHead, read, dump, sha, key, digest

def save(p, **v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **v)
    tmp.replace(p)

def snapshot(p, v):
    tmp = Path(p).with_suffix('.tmp')
    torch.save(v, tmp)
    tmp.replace(p)

def fingerprint(model):
    return digest({n: p.detach().reshape(-1)[torch.linspace(0, p.numel() - 1, min(p.numel(), 16), device=p.device).long()].float().cpu() for n, p in model.named_parameters() if p.numel()})

def load_native(assets, out):
    from sam_3d_body import SAM3DBodyEstimator
    import sam_3d_body.build_models as builder
    from sam_body_feasibility import audit_keys, DINO_REVISION
    if not getattr(torch.hub.load, 'r147_pinned', False):
        original = torch.hub.load

        def pinned(repo, *a, **kw):
            return original(repo + ':' + DINO_REVISION if repo == 'facebookresearch/dinov3' else repo, *a, **kw)
        pinned.r147_pinned = True
        torch.hub.load = pinned
    old = builder.load_state_dict

    def audited(model, state_dict, strict=False, **kw):
        values = model.state_dict()
        a = audit_keys(values, state_dict)
        assert not a['unsafe_missing'] and (not a['unexpected']), a
        for n in a['initialized_only']:
            expected = torch.zeros_like(values[n]) if n.endswith('mask_token') else torch.eye(54, device=values[n].device, dtype=values[n].dtype)
            assert torch.equal(values[n], expected), n
        dump(out / 'checkpoint_load.json', a)
        return old(model, state_dict, strict=False, **kw)
    builder.load_state_dict = audited
    try:
        model, cfg = builder.load_sam_3d_body(str(assets / 'model.ckpt'), device='cuda', mhr_path=str(assets / 'mhr_model.pt'))
    finally:
        builder.load_state_dict = old
    model.eval().requires_grad_(False)
    assert cfg.MODEL.DECODER.USE_INTRIN_CENTER and cfg.MODEL.DECODER.DIM == 1024
    assert all((not p.requires_grad for p in model.parameters())) and all((not m.training for m in model.modules()))
    return (model, SAM3DBodyEstimator(model, cfg))

def infer(model, est, root, row, frame):
    import cv2
    image = cv2.imread(str(root / frame['file']))
    assert image.shape == (384, 384, 3) and sha(root / frame['file']) == frame['image_sha256']
    if row['flip']:
        image = image[:, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    captured = []
    body = []
    hand = []

    def body_hook(module, args):
        body.append(args[0].detach())

    def pose_hook(module, args):
        x = args[0]
        assert x.is_cuda and x.shape == (1, 1024) and torch.equal(x, body[-1])
        captured.append(x.detach().float().cpu().numpy()[0].copy())
    handles = [model.head_pose.register_forward_pre_hook(body_hook), model.head_pose.proj.register_forward_pre_hook(pose_hook), model.head_pose_hand.proj.register_forward_pre_hook(lambda m, a: hand.append(True))]
    torch.cuda.synchronize()
    start = time.monotonic()
    try:
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            result = est.process_one_image(image, bboxes=np.array(frame['bbox'], np.float32)[None], cam_int=torch.tensor(frame['intrinsic'], dtype=torch.float32, device='cuda')[None], inference_type='body')
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
    finally:
        for h in handles:
            h.remove()
    assert len(result) == 1 and captured and (len(captured) == len(body)) and (not hand)
    token = captured[-1]
    assert token.shape == (1024,) and np.isfinite(token).all() and (token.std() > 1e-05)
    r = result[0]
    xyz = np.asarray(r['pred_keypoints_3d']) + np.asarray(r['pred_cam_t'])
    h = xyz @ np.asarray(frame['intrinsic']).T
    uv = h[:, :2] / h[:, 2:]
    error = float(abs(uv - np.asarray(r['pred_keypoints_2d'])).max())
    assert error < 0.02
    return (token, dict(seconds=seconds, body_projection_calls=len(captured), hand_projection_calls=len(hand), hook='head_pose.proj input,last invocation', device='cuda', projection_error=error))

def lr_value(step, steps):
    warm = max(1, int(0.1 * steps))
    return (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm - 1, 1)))
