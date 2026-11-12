"""Compact modality-conditioned U-Net and local MPS training.

No pretrained weights or test labels are used. Sampling is subject-balanced.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from .data import read_rows, read_image, image_path, parse_instances, split_id, write_rows
from .rle import Mask, labels_to_instances
from .segmentation import normalize


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU())


class SmallUNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.a = Block(3, width)
        self.b = Block(width, width*2)
        self.c = Block(width*2, width*4)
        self.d = Block(width*6, width*2)
        self.e = Block(width*3, width)
        self.out = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        return self.out(self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1)))


def image_features(image):
    raw = normalize(image)
    local = np.maximum(raw-ndi.gaussian_filter(raw, 10), 0)
    high = np.percentile(local, 99.5)
    local = np.clip(local/max(high, 1e-5), 0, 1)
    return np.stack([raw, local]).astype(np.float32)


def target_maps(instances, shape):
    labels = np.zeros(shape, np.int32)
    for n, value in enumerate(instances.values(), 1):
        parsed = Mask.parse(value, shape)
        for start, end in zip(parsed.starts, parsed.ends):
            labels.ravel()[start:end] = n
    centers = np.zeros(shape, np.float32)
    for n, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        obj = labels[box] == n
        distance = ndi.distance_transform_edt(np.pad(obj, 1))[1:-1, 1:-1]
        centers[box] = np.maximum(centers[box], distance/max(float(distance.max()), 1))
    return np.stack([labels > 0, centers]).astype(np.float32), labels


def load_training(root, held_out):
    rows = read_rows(Path(root)/"training/train_ground_truth.csv")
    subjects = sorted({split_id(sid)[0] for sid in rows})
    if held_out not in subjects:
        raise ValueError("Held-out subject is not present in training data")
    items, seen = [], set()
    train_ids = []
    for sid, row in rows.items():
        subject = split_id(sid)[0]
        if subject == held_out:
            continue
        train_ids.append(sid)
        for mode, modality in enumerate(("invivo", "exvivo")):
            image = read_image(image_path(root, "training", sid, modality))
            # Avoid exactly duplicated image+annotation samples. Different
            # annotation sets are retained and explicitly remain same-subject.
            key = hashlib.sha256(image.tobytes()+json.dumps(row[f"{modality}_instances"], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            target, labels = target_maps(row[f"{modality}_instances"], image.shape)
            centers = np.array(ndi.center_of_mass(labels > 0, labels, np.unique(labels)[1:])).reshape(-1, 2)
            items.append({"x": image_features(image).astype(np.float16), "y": target.astype(np.float16),
                          "centers": centers, "mode": mode, "subject": subject, "sample_id": sid})
    return items, train_ids


def sample_batch(items, batch, patch, rng):
    groups = sorted({(r["subject"], r["mode"]) for r in items})
    xs, ys = [], []
    for _ in range(batch):
        subject, mode = groups[int(rng.integers(len(groups)))]
        candidates = [r for r in items if r["subject"] == subject and r["mode"] == mode]
        record = candidates[int(rng.integers(len(candidates)))]
        height, width = record["x"].shape[1:]
        if rng.random() < 0.7 and len(record["centers"]):
            center = record["centers"][int(rng.integers(len(record["centers"])))]
            cy, cx = center + rng.uniform(-patch/3, patch/3, 2)
        else:
            cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        y = int(np.clip(cy-patch/2, 0, max(0, height-patch)))
        x = int(np.clip(cx-patch/2, 0, max(0, width-patch)))
        feature = record["x"][:, y:y+patch, x:x+patch].astype(np.float32)
        target = record["y"][:, y:y+patch, x:x+patch].astype(np.float32)
        padding = ((0, 0), (0, patch-feature.shape[1]), (0, patch-feature.shape[2]))
        feature = np.pad(feature, padding, mode="reflect")
        target = np.pad(target, padding, mode="constant")
        k = int(rng.integers(4))
        feature, target = np.rot90(feature, k, (1, 2)), np.rot90(target, k, (1, 2))
        if rng.random() < 0.5:
            feature, target = feature[:, :, ::-1], target[:, :, ::-1]
        feature = np.clip(feature**rng.uniform(0.7, 1.4)*rng.uniform(0.8, 1.2) + rng.normal(0, 0.01, feature.shape), 0, 1).astype(np.float32)
        xs.append(np.concatenate([feature, np.full((1, patch, patch), mode, np.float32)]))
        ys.append(target)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def loss_function(logits, targets):
    fg, center = targets[:, :1], targets[:, 1:]
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits[:, :1], fg, pos_weight=torch.tensor(2., device=logits.device))
    dims = (1, 2, 3)
    dice = 1 - ((2*(probabilities[:, :1]*fg).sum(dims)+1)/(probabilities[:, :1].sum(dims)+fg.sum(dims)+1)).mean()
    distance = (F.smooth_l1_loss(probabilities[:, 1:], center, reduction="none")*(1+4*fg)).mean()
    return bce + dice + 2*distance


def choose_device(request):
    if request == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process. Run from a normal terminal or permit local GPU access.")
    return request


def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"model.pt").exists():
        raise FileExistsError("Choose a new experiment output directory; existing checkpoints are preserved")
    items, train_ids = load_training(args.data, args.held_out)
    assert all(r["subject"] != args.held_out for r in items)
    model = SmallUNet(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr/10)
    rng = np.random.default_rng(args.seed)
    config = vars(args).copy()
    config.update(device=device, train_ids=train_ids, parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"training_config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(f"Training {config['parameters']:,} parameters on {device}; {len(train_ids)} regions; held out {args.held_out}", flush=True)
    model.train()
    start = time.perf_counter()
    history = []
    for step in range(1, args.steps+1):
        x, y = sample_batch(items, args.batch_size, args.patch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x.to(device)), y.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        schedule.step()
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite training loss")
        history.append({"step": step, "loss": value, "seconds": time.perf_counter()-start})
        if step % 50 == 0 or step in (1, args.steps):
            print(json.dumps(history[-1]), flush=True)
        if step % 500 == 0 or step == args.steps:
            state = {"state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()}, "width": args.width,
                     "step": step, "held_out": args.held_out, "train_ids": train_ids, "config": config}
            temp = out/"model.tmp"
            torch.save(state, temp)
            temp.replace(out/"model.pt")
            (out/"history.json").write_text(json.dumps(history, indent=2)+"\n")
    print(f"Saved {out/'model.pt'}", flush=True)


@torch.inference_mode()
def probability_maps(model, image, mode, device, tile=256, overlap=32, batch_size=4):
    features = image_features(image)
    height, width = image.shape
    def starts(size):
        if size <= tile:
            return [0]
        return sorted(set(list(range(0, size-tile+1, tile-overlap)) + [size-tile]))
    positions = [(y,x) for y in starts(height) for x in starts(width)]
    accumulated = np.zeros((2, height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 0.01).astype(np.float32)
    for offset in range(0, len(positions), batch_size):
        batch_positions = positions[offset:offset+batch_size]
        batch = []
        for y,x in batch_positions:
            crop = features[:, y:y+tile, x:x+tile]
            crop = np.pad(crop, ((0,0),(0,tile-crop.shape[1]),(0,tile-crop.shape[2])), mode="reflect")
            batch.append(np.concatenate([crop, np.full((1,tile,tile),mode,np.float32)]))
        predictions = model(torch.from_numpy(np.stack(batch)).to(device)).sigmoid().cpu().numpy()
        for (y,x), prediction in zip(batch_positions, predictions):
            h,w = min(tile,height-y),min(tile,width-x)
            accumulated[:,y:y+h,x:x+w] += prediction[:,:h,:w]*window[:h,:w]
            weights[y:y+h,x:x+w] += window[:h,:w]
    return accumulated/weights


def instances_from_maps(maps, threshold=0.5, seed_threshold=0.35, min_area=12):
    foreground = maps[0] > threshold
    score = ndi.gaussian_filter(maps[1], 0.7)
    coordinates = peak_local_max(score, min_distance=3, threshold_abs=seed_threshold,
                                 labels=foreground.astype(np.uint8), exclude_border=False)
    markers = np.zeros(foreground.shape, np.int32)
    if not len(coordinates):
        return markers
    markers[tuple(coordinates.T)] = np.arange(1,len(coordinates)+1)
    labels = watershed(-score,markers,mask=foreground)
    sizes = np.bincount(labels.ravel())
    keep = (sizes >= min_area) & (sizes <= 600)
    keep[0] = False
    remap = np.zeros(len(sizes), np.int32)
    remap[keep] = np.arange(1,keep.sum()+1)
    return remap[labels]


def predict(args):
    torch.set_num_threads(4)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SmallUNet(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    root = Path(args.data)
    source = root/("training/train_ground_truth.csv" if args.split=="training" else "sample_submission.csv")
    rows = read_rows(source)
    selected = [sid for sid in rows if args.subject is None or split_id(sid)[0]==args.subject]
    if not selected:
        raise ValueError("No selected regions")
    if args.split=="training" and not args.allow_training_subjects:
        if any(sid in checkpoint["train_ids"] for sid in selected):
            raise ValueError("Evaluation includes training samples. Select the held-out subject or explicitly request development predictions.")
    out = Path(args.output)
    cache = out.parent/(out.stem+"_maps")
    cache.mkdir(parents=True,exist_ok=True)
    predictions = []
    started = time.perf_counter()
    for n,sid in enumerate(selected,1):
        row={"sample_id":sid,"match_pairs":[]}
        for mode,modality in enumerate(("invivo","exvivo")):
            image=read_image(image_path(root,args.split,sid,modality))
            maps=probability_maps(model,image,mode,device)
            np.savez_compressed(cache/f"{sid}_{modality}.npz",maps=maps.astype(np.float16))
            labels=instances_from_maps(maps,args.threshold,args.seed_threshold)
            row[f"{modality}_instances"]=labels_to_instances(labels,"IVP" if mode==0 else "EXP")
        predictions.append(row)
        print(f"[{n}/{len(selected)}] {sid}: {len(row['invivo_instances'])}/{len(row['exvivo_instances'])} cells",flush=True)
    write_rows(predictions,out)
    manifest=vars(args).copy()
    manifest.update(device=device,held_out=checkpoint['held_out'],train_ids=checkpoint['train_ids'],seconds=time.perf_counter()-started,
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    Path(str(out)+".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    t=sub.add_parser("train")
    t.add_argument("--data",required=True)
    t.add_argument("--held-out",required=True)
    t.add_argument("--output",required=True)
    t.add_argument("--steps",type=int,default=1500)
    t.add_argument("--batch-size",type=int,default=8)
    t.add_argument("--patch-size",type=int,default=128)
    t.add_argument("--width",type=int,default=16)
    t.add_argument("--lr",type=float,default=0.001)
    t.add_argument("--seed",type=int,default=2026)
    t.add_argument("--device",choices=("auto","mps","cpu"),default="auto")
    p=sub.add_parser("predict")
    p.add_argument("--data",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--split",choices=("training","hidden_test"),default="training")
    p.add_argument("--subject")
    p.add_argument("--allow-training-subjects",action="store_true")
    p.add_argument("--output",required=True)
    p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--seed-threshold",type=float,default=0.35)
    p.add_argument("--device",choices=("auto","mps","cpu"),default="auto")
    args=parser.parse_args()
    train(args) if args.action=="train" else predict(args)


if __name__=="__main__":
    main()
