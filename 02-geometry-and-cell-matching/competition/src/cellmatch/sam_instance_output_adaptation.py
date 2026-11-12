"""Frozen SAM spatial features with trainable instance mask and IoU output MLPs."""
import hashlib
import numpy as np
import torch
from segment_anything import sam_model_registry
from cellmatch.direct_center_ex import sha
from cellmatch.sam_lora_continuation import MODEL, field

PREFIXES = tuple('mask_decoder.output_hypernetworks_mlps.%d.' % i for i in (1, 2, 3)) + ('mask_decoder.iou_prediction_head.',)

def load(device='cpu'):
    assert sha(MODEL) == 'ffe862fa5b58d78375506041faf0eb3927ced5b5391e8be383e6d7ac07aed924'
    sam = sam_model_registry['vit_b']()
    sam.load_state_dict(torch.load(MODEL, map_location='cpu', weights_only=True), strict=True)
    sam.requires_grad_(False)
    for name, parameter in sam.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    assert sum(p.numel() for p in sam.parameters() if p.requires_grad) == 552036
    return sam.to(device).eval()

def frozen_digest(sam):
    digest = hashlib.sha256()
    for name, value in sorted(sam.state_dict().items()):
        if name.startswith(PREFIXES):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()

def prompts(sam, points):
    device = next(sam.parameters()).device
    query = torch.as_tensor(np.asarray(points, dtype=np.float32) * 4, device=device)[:, None]
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=(query, torch.ones((len(points), 1), dtype=torch.long, device=device)), boxes=None, masks=None)
    return dict(sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                image_pe=sam.prompt_encoder.get_dense_pe())

def decode(sam, features, points, all_tokens=False):
    decoder = sam.mask_decoder
    prompt = prompts(sam, points)
    # Recompute point-dependent features; no inference_mode tensors enter autograd.
    with torch.no_grad():
        output_tokens = torch.cat((decoder.iou_token.weight, decoder.mask_tokens.weight), dim=0)
        output_tokens = output_tokens[None].expand(len(points), -1, -1)
        tokens = torch.cat((output_tokens, prompt['sparse_prompt_embeddings']), dim=1)
        src = torch.repeat_interleave(features, len(points), dim=0) + prompt['dense_prompt_embeddings']
        pos = torch.repeat_interleave(prompt['image_pe'], len(points), dim=0)
        batch, channels, height, width = src.shape
        hs, src = decoder.transformer(src, pos, tokens)
        upscaled = decoder.output_upscaling(src.transpose(1, 2).view(batch, channels, height, width))
    ids = (0, 1, 2, 3) if all_tokens else (1, 2, 3)
    hyper = torch.stack([decoder.output_hypernetworks_mlps[i](hs[:, i + 1]) for i in ids], dim=1)
    batch, channels, height, width = upscaled.shape
    low = (hyper @ upscaled.view(batch, channels, height * width)).view(batch, len(ids), height, width)
    quality = decoder.iou_prediction_head(hs[:, 0])[:, list(ids)]
    return sam.postprocess_masks(low, (1024, 1024), (256, 256)), quality

def objective(logits, quality, targets):
    truth = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=logits.device)[:, None]
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * truth).sum((-2, -1)) + 1e-6) / (probability.sum((-2, -1)) + truth.sum((-2, -1)) + 1e-6)
    best, winner = dice.min(dim=1)
    with torch.no_grad():
        binary = logits > 0; target_binary = truth > 0
        actual_iou = (binary & target_binary).sum((-2, -1)).float() / (binary | target_binary).sum((-2, -1)).clamp_min(1)
    mask_loss = best.mean()
    quality_loss = torch.nn.functional.mse_loss(quality, actual_iou)
    return mask_loss + quality_loss, mask_loss, quality_loss, winner

def metrics(logits, quality, targets):
    logits = np.asarray(logits); quality = np.asarray(quality)
    chosen = quality.argmax(axis=1)
    prediction = logits[np.arange(len(logits)), chosen] > 0
    truth = np.asarray(targets) > 0
    iou = (prediction & truth).sum((1, 2)) / np.maximum((prediction | truth).sum((1, 2)), 1)
    return dict(selected_tokens=(chosen + 1).tolist(), ious=iou.tolist(),
                strict_valid=int((iou > .75).sum()), mean_iou=float(iou.mean()))
