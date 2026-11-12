"""Native32 contours conditioned on a native128 image and predicted neighbors."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from .learning import SmallUNet, Block, image_features
from .data import parse_instances
from .ex_boundary import origin, local_mask, patch_rle
from .rle import Mask


class NeighborhoodUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct the original model first: seed2083 retains its exact initializer.
        self.base = SmallUNet(16)
        self.context = nn.ModuleList([Block(3, 8), Block(8, 16),
                                      Block(16, 24), Block(24, 32)])
        self.projection = nn.Conv2d(64, 16, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        assert x.shape[1:] == (4, 128, 128)
        local = x[:, :3, 48:80, 48:80]
        context = x[:, [0, 1, 3]]
        for index, block in enumerate(self.context):
            context = block(F.avg_pool2d(context, 2) if index else context)
        central = F.interpolate(context, size=(128, 128), mode="bilinear",
                                align_corners=False)[:, :, 48:80, 48:80]
        global_context = context.mean((-2, -1), keepdim=True).expand(-1, -1, 32, 32)
        a = self.base.a(local) + self.projection(torch.cat([central, global_context], 1))
        b = self.base.b(F.avg_pool2d(a, 2))
        c = self.base.c(F.avg_pool2d(b, 2))
        d = self.base.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear",
                                               align_corners=False), b], 1))
        e = self.base.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear",
                                               align_corners=False), a], 1))
        return self.base.out(e)


class CentralOnly(nn.Module):
    """Exact original-model control using the packed context input interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, :3, 48:80, 48:80])


def crop128(array, x0, y0):
    sx, sy = x0 - 48, y0 - 48
    h, w = array.shape[-2:]
    output = np.zeros((*array.shape[:-2], 128, 128), array.dtype)
    x1, x2, y1, y2 = max(0, sx), min(w, sx + 128), max(0, sy), min(h, sy + 128)
    if x2 > x1 and y2 > y1:
        output[..., y1-sy:y2-sy, x1-sx:x2-sx] = array[..., y1:y2, x1:x2]
    return output


def pack_cached(context, raw32):
    output = np.zeros((4, 128, 128), np.float32)
    output[:2] = context[:2]
    output[2, 48:80, 48:80] = raw32[2]
    output[3] = context[2]
    assert np.array_equal(output[:3, 48:80, 48:80], raw32.astype(np.float32))
    return output


def pack_query(features, source_union, query, x0, y0):
    output = np.zeros((4, 128, 128), np.float32)
    output[:2] = crop128(features, x0, y0)
    output[2, 48:80, 48:80] = local_mask(query, x0, y0)
    output[3] = crop128(source_union, x0, y0)
    xy = query.coordinates() - [x0 - 48, y0 - 48]
    xy = xy[((xy >= 0) & (xy < 128)).all(1)]
    output[3, xy[:, 1], xy[:, 0]] = 0
    return output


def refine_row(base_row, source_row, image, model, device, scope):
    """Original guarded native operator; context input and explicit scope only."""
    assert scope in ("paired", "unpaired")
    features = image_features(image)
    row = copy.deepcopy(base_row)
    masks = parse_instances(row, 'exvivo', image.shape)
    source = parse_instances(source_row, 'exvivo', image.shape)
    union = np.zeros(image.shape, bool)
    for mask in source.values():
        for start, end in zip(mask.starts, mask.ends):
            union.ravel()[start:end] = True
    paired = {e for _, e in row['match_pairs']}
    rles = set(source_row['exvivo_instances'].values())
    keys = [k for k in masks if ((k in paired) == (scope == 'paired'))
            and row['exvivo_instances'][k] in rles]
    occupied = np.zeros(image.shape, np.int32)
    numbers = {k: n for n, k in enumerate(masks, 1)}
    for k, mask in masks.items():
        for start, end in zip(mask.starts, mask.ends):
            occupied.ravel()[start:end] = numbers[k]
    accepted = []
    # 16 rather than64 bounds128-context inference memory; no batch-dependent layers.
    for offset in range(0, len(keys), 16):
        batchkeys = keys[offset:offset+16]
        positions = [origin(masks[k]) for k in batchkeys]
        patches = [pack_query(features, union, masks[k], *xy)
                   for k, xy in zip(batchkeys, positions)]
        with torch.inference_mode():
            probs = model(torch.from_numpy(np.stack(patches)).to(device))[:, 0].sigmoid().cpu().numpy()
        for key, (x0, y0), probability in zip(batchkeys, positions, probs):
            old = masks[key]
            prior = local_mask(old, x0, y0).astype(bool)
            labels, count = ndi.label(probability > .5)
            overlap = np.bincount(labels[prior], minlength=count+1)
            overlap[0] = 0
            candidate = labels == int(overlap.argmax()) if overlap.max() else np.zeros_like(prior)
            block = occupied[y0:y0+32, x0:x0+32]
            candidate &= (block == 0) | (block == numbers[key])
            encoded = patch_rle(candidate, x0, y0, image.shape)
            new = Mask.parse(encoded, image.shape)
            if new.area and .5 <= new.area/old.area <= 1.7 and new.iou(old) > .4:
                row['exvivo_instances'][key] = encoded
                block[candidate] = numbers[key]
                accepted.append(key)
    assert row['match_pairs'] == base_row['match_pairs']
    assert row['invivo_instances'] == base_row['invivo_instances']
    assert set(row['exvivo_instances']) == set(base_row['exvivo_instances'])
    assert set(accepted) <= set(keys)
    taken = np.zeros(image.shape, bool)
    for mask in parse_instances(row, 'exvivo', image.shape).values():
        xy = mask.coordinates()
        assert not taken[xy[:, 1], xy[:, 0]].any()
        taken[xy[:, 1], xy[:, 0]] = True
    return row, dict(sample_id=base_row['sample_id'], eligible=keys, accepted=accepted,
                     changed=sum(row['exvivo_instances'][k] != base_row['exvivo_instances'][k] for k in keys))
