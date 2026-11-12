"""Shared pretrained RGB encoder + masked three-view GRU + temporal attention.

Stages: download, encode, train, predict. This is our reference implementation,
not a reproduction of the unavailable teacher YOLO/GRU baseline.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import rgb_gru_data as data

PRETRAINED = EfficientNet_B0_Weights.IMAGENET1K_V1
ENCODER_CONFIG = {"model": "torchvision.efficientnet_b0", "weights": "IMAGENET1K_V1",
                  "url": PRETRAINED.url, "feature_dim": 1280,
                  "normalization": "ImageNet mean/std; shared across three views"}
MODEL_CONFIG = {"feature_dim": 1280, "projection": 128, "hidden": 128, "dropout": 0.2}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_encoder(weights_dir, download=False):
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / PRETRAINED.url.rsplit("/", 1)[-1]
    if not path.exists() and not download:
        raise FileNotFoundError("Run the download stage once; training never downloads implicitly")
    if not path.exists():
        torch.hub.download_url_to_file(PRETRAINED.url, str(path), hash_prefix="7f5810bc", progress=True)
    digest = data.sha(path)
    if not digest.startswith("7f5810bc"):
        raise ValueError("Official weight checksum mismatch")
    manifest = {**ENCODER_CONFIG, "sha256": digest, "bytes": path.stat().st_size}
    data.immutable_json(weights_dir / "manifest.json", manifest)
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.classifier = nn.Identity()
    return model, manifest


def normalize(images):
    # Input shape N,C,H,W. Missing views are excluded before encoding.
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return (images / 255.0 - mean) / std


def image_features(encoder, images, mask, chunk=8):
    b, t, v, c, h, w = images.shape
    flat = images.reshape(-1, c, h, w)
    positions = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    result = images.new_zeros((b * t * v, ENCODER_CONFIG["feature_dim"]))
    if len(positions):
        values = torch.cat([encoder(normalize(flat[part])) for part in positions.split(chunk)], dim=0)
        result = result.index_copy(0, positions, values)
    return result.reshape(b, t, v, -1)


class TemporalModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or MODEL_CONFIG)
        c = self.config
        self.project = nn.Sequential(nn.Linear(c["feature_dim"], c["projection"]),
                                     nn.LayerNorm(c["projection"]), nn.GELU())
        self.gru = nn.GRU(c["projection"] * 3 + 3 + 2, c["hidden"], batch_first=True)
        self.attention = nn.Sequential(nn.Linear(c["hidden"], 64), nn.Tanh(), nn.Linear(64, 1))
        self.dropout = nn.Dropout(c["dropout"])
        self.heads = nn.ModuleDict({name: nn.Linear(c["hidden"] + 3, len(labels))
                                    for name, labels in data.TARGETS.items()})

    def forward(self, features, view_mask, times, lengths, metadata):
        b, t, _, _ = features.shape
        valid_time = torch.arange(t, device=features.device)[None] < lengths.to(features.device)[:, None]
        mask = view_mask & valid_time[:, :, None]
        # masked_fill, rather than NaN * 0, prevents missing observations leaking.
        clean = features.masked_fill(~mask[:, :, :, None], 0)
        projected = self.project(clean).masked_fill(~mask[:, :, :, None], 0).flatten(2)
        end = times.gather(1, (lengths.to(times.device) - 1)[:, None])
        relative = (times - end).clamp(-10, 0) / 2
        delta = torch.cat([torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]], dim=1)
        timing = torch.stack([relative, delta.clamp(0, 10)], dim=-1)
        x = torch.cat([projected, mask.float(), timing], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=t)
        attention = self.attention(sequence).squeeze(-1).masked_fill(~valid_time, float("-inf")).softmax(1)
        pooled = (sequence * attention[:, :, None]).sum(1)
        pooled = torch.cat([self.dropout(pooled), metadata], dim=1)
        return {key: head(pooled) for key, head in self.heads.items()}, attention


class MeanTemporalModel(nn.Module):
    """Order-invariant control using exactly the same frozen visual features."""
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or MODEL_CONFIG)
        c = self.config
        self.project = nn.Sequential(nn.Linear(c['feature_dim'], c['projection']),
                                     nn.LayerNorm(c['projection']), nn.GELU())
        self.dropout = nn.Dropout(c['dropout'])
        self.heads = nn.ModuleDict({k: nn.Linear(c['projection'] * 3 + 3, len(v))
                                    for k, v in data.TARGETS.items()})

    def forward(self, features, view_mask, times, lengths, metadata):
        valid = torch.arange(features.shape[1], device=features.device)[None] < lengths.to(features.device)[:, None]
        mask = view_mask & valid[:, :, None]
        clean = features.masked_fill(~mask[..., None], 0)
        x = self.project(clean).masked_fill(~mask[..., None], 0)
        pooled = x.sum(1) / mask.sum(1).clamp_min(1)[..., None]
        pooled = torch.cat([self.dropout(pooled.flatten(1)), metadata], 1)
        attention = valid.float() / lengths.to(features.device)[:, None]
        return {k: head(pooled) for k, head in self.heads.items()}, attention


class Predictor(nn.Module):
    def __init__(self, encoder=None, config=None, finetune=False):
        super().__init__()
        cls = MeanTemporalModel if (config or {}).get('aggregation') == 'mean' else TemporalModel
        self.temporal = cls(config)
        self.encoder = encoder
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = False
            if finetune:
                for p in encoder.features[-2:].parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            # Tiny clip batches must not overwrite pretrained BN statistics.
            self.encoder.eval()
        return self

    def forward(self, batch):
        x = batch["inputs"]
        if self.encoder is not None:
            x = image_features(self.encoder, x, batch["view_mask"])
        return self.temporal(x, batch["view_mask"], batch["timestamps"], batch["lengths"], batch["metadata"])


class ClipDataset(Dataset):
    def __init__(self, rows, cache, mode="features", views="all"):
        self.rows = rows.reset_index(drop=True)
        self.cache, self.mode, self.views = Path(cache), mode, views

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        folder = "features" if self.mode == "features" else "crops"
        path = self.cache / folder / (row["clip"] + ".npz")
        with np.load(path, allow_pickle=False) as a:
            inputs = a["features"].astype(np.float32) if self.mode == "features" else a["images"].transpose(0, 1, 4, 2, 3).astype(np.float32)
            mask = a["view_mask"].copy()
            times = a["timestamps"].astype(np.float32)
        if self.views == "scene":
            mask[:, 1:] = False
        elif self.views == "players":
            mask[:, 0] = False
        metadata = np.zeros(3, np.float32)
        metadata[{"left": 0, "right": 1}.get(row.get("preferred_foot"), 2)] = 1
        item = {"clip": row["clip"], "inputs": torch.from_numpy(inputs),
                "view_mask": torch.from_numpy(mask), "timestamps": torch.from_numpy(times),
                "metadata": torch.from_numpy(metadata)}
        if all(name in row.index for name in data.TARGETS):
            item["labels"] = torch.tensor([values.index(row[name]) for name, values in data.TARGETS.items()])
        return item


def collate(items):
    lengths = torch.tensor([len(x["timestamps"]) for x in items])
    result = {"clips": [x["clip"] for x in items], "lengths": lengths,
              "metadata": torch.stack([x["metadata"] for x in items])}
    for key in ["inputs", "view_mask", "timestamps"]:
        result[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in items], batch_first=True)
    if "labels" in items[0]:
        result["labels"] = torch.stack([x["labels"] for x in items])
    return result


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) and key != "lengths" else value
            for key, value in batch.items()}


def loader(rows, args, mode, shuffle=False):
    return DataLoader(ClipDataset(rows, args.cache, mode, args.views), batch_size=args.batch_size,
                      shuffle=shuffle, collate_fn=collate, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))


def loss_for(logits, labels):
    return sum(weight * nn.functional.cross_entropy(logits[key], labels[:, i])
               for i, (key, weight) in enumerate(zip(data.TARGETS, data.WEIGHTS)))


def metric_report(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    accuracies = (truth == predictions).mean(axis=0)
    result = {key: float(accuracies[i]) for i, key in enumerate(data.TARGETS)}
    result["weighted_score"] = float(accuracies @ np.array(data.WEIGHTS))
    result["clips"] = len(truth)
    result["confusion_matrices"] = {}
    for i, (key, values) in enumerate(data.TARGETS.items()):
        matrix = np.zeros((len(values), len(values)), dtype=int)
        np.add.at(matrix, (truth[:, i], predictions[:, i]), 1)
        result["confusion_matrices"][key] = {"labels": values, "rows_truth_columns_pred": matrix.tolist()}
    return result


def warm_start_temporal(model, checkpoint, rows, args, encoder_manifest):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    previous = saved["config"]
    if (previous['model'].get('aggregation', 'gru') != getattr(args, 'aggregation', 'gru')
            or previous["finetune"] or previous["views"] != args.views
            or previous["fold"] != args.fold or previous["data"] != data.CONFIG
            or previous["encoder"] != encoder_manifest
            or previous["smoke_only"] != args.smoke
            or set(previous["train_clips"]) != set(rows.loc[rows.fold != args.fold, "clip"])
            or set(previous["validation_clips"]) != set(rows.loc[rows.fold == args.fold, "clip"])
            or previous["fold_csv_sha256"] != data.sha(args.cache / "public_folds.csv")):
        raise ValueError("Warm start must use the matching frozen-encoder experiment and split")
    provenance = validate_caches(rows, args.cache, "features")
    if provenance != previous["input_provenance"]:
        raise ValueError("Warm-start inputs differ from the frozen-encoder experiment")
    state = {key.removeprefix("temporal."): value for key, value in saved["state_dict"].items()
             if key.startswith("temporal.")}
    model.temporal.load_state_dict(state, strict=True)


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    records, truth, predictions, total_loss, count = [], [], [], 0., 0
    for batch in batches:
        batch = to_device(batch, device)
        logits, attention = model(batch)
        pred = torch.stack([logits[key].argmax(1) for key in data.TARGETS], dim=1)
        if "labels" in batch:
            total_loss += float(loss_for(logits, batch["labels"])) * len(pred)
            truth.extend(batch["labels"].cpu().tolist())
        predictions.extend(pred.cpu().tolist())
        count += len(pred)
        for j, clip in enumerate(batch["clips"]):
            n = int(batch["lengths"][j])
            row = {"clip": clip,
                   "attention": attention[j, :n].cpu().tolist(),
                   "timestamps": batch["timestamps"][j, :n].cpu().tolist()}
            for i, (key, labels) in enumerate(data.TARGETS.items()):
                row[key] = labels[int(pred[j, i])]
                row[key + "_probabilities"] = logits[key][j].softmax(-1).cpu().tolist()
            records.append(row)
    metrics = metric_report(truth, predictions) if truth else {}
    if truth:
        metrics["loss"] = total_loss / count
    return metrics, records


def validate_caches(rows, cache, mode):
    """Verify on-disk hashes once before a run, not inside every training batch."""
    provenance = {}
    for clip in rows["clip"]:
        crop = cache / "crops" / (clip + ".npz")
        meta = json.loads(crop.with_suffix(".json").read_text())
        if meta["cache_sha256"] != data.sha(crop) or meta["inputs"]["config"] != data.CONFIG:
            raise ValueError(f"Invalid crop cache: {clip}")
        if mode == "features":
            path = cache / "features" / (clip + ".npz")
            feature_meta = json.loads(path.with_suffix(".json").read_text())
            if feature_meta["crop_sha256"] != meta["cache_sha256"] or feature_meta["feature_sha256"] != data.sha(path):
                raise ValueError(f"Invalid feature cache: {clip}")
            if feature_meta["encoder"] != json.loads((cache / "encoder_manifest.json").read_text()):
                raise ValueError(f"Mixed pretrained feature encoders: {clip}")
        provenance[clip] = {"crop_sha256": meta["cache_sha256"],
                            "track_sha256": meta["inputs"]["track_sha256"],
                            "view_coverage": meta["view_coverage"]}
    return provenance


def encode(args):
    started = time.monotonic()
    encoder, manifest = make_encoder(args.weights)
    encoder.eval().to(args.device)
    data.immutable_json(args.cache / "encoder_manifest.json", manifest)
    paths = sorted((args.cache / "crops").glob("*.npz"))
    if not paths:
        raise ValueError("No crop caches; run rgb_gru_data.py cache first")
    for path in paths:
        clip = path.name.removesuffix(".npz")
        validate_caches(pd.DataFrame({"clip": [clip]}), args.cache, "images")
        dest = args.cache / "features" / path.name
        metadata = {"encoder": manifest, "crop_sha256": data.sha(path)}
        if dest.exists() and dest.with_suffix(".json").exists():
            old = json.loads(dest.with_suffix(".json").read_text())
            if all(old[k] == v for k, v in metadata.items()) and old["feature_sha256"] == data.sha(dest):
                continue
            raise ValueError(f"Stale encoded features: {clip}")
        with np.load(path, allow_pickle=False) as a:
            images = torch.from_numpy(a["images"].transpose(0, 1, 4, 2, 3).copy()).float()[None].to(args.device)
            mask = torch.from_numpy(a["view_mask"])[None].to(args.device)
            with torch.inference_mode():
                features = image_features(encoder, images, mask)[0].cpu().numpy()
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, features=features, view_mask=a["view_mask"],
                                timestamps=a["timestamps"], frame_indices=a["frame_indices"])
        data.dump(dest.with_suffix(".json"), {**metadata, "feature_sha256": data.sha(dest)})
        print(f"Encoded {clip}: {features.shape}", flush=True)
        if getattr(args, 'max_seconds', None) and time.monotonic() - started >= args.max_seconds:
            print('Feature checkpoint saved; time budget reached.', flush=True)
            return False
    return True


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    name, values, pos, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def restore_training(path, model, optimizer, generator, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    # Device/library versions are recorded, but a checkpoint may move hosts.
    comparable = lambda c: {k: v for k, v in c.items() if k not in {'device', 'torch'}}
    if comparable(saved['config']) != comparable(config):
        raise ValueError('Resume configuration/data/code mismatch')
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'], generator)
    return saved


def train(args):
    started = time.monotonic()
    resume = getattr(args, 'resume', False)
    if args.run.exists() and not resume:
        raise ValueError("Use a new --run directory; existing experiments are immutable")
    if resume and not (args.run / 'last.pt').exists():
        raise ValueError('Resume requires a saved last.pt checkpoint')
    mode = "images" if args.finetune else "features"
    rows = pd.read_csv(args.cache / "public_folds.csv")
    if args.smoke:
        folder = "crops" if args.finetune else "features"
        rows = rows[rows["clip"].map(lambda c: (args.cache / folder / (c + ".npz")).exists())]
    provenance = validate_caches(rows, args.cache, mode)
    if not args.smoke and args.views != "scene":
        unprocessed = [clip for clip, p in provenance.items() if p["track_sha256"] is None]
        if unprocessed:
            raise ValueError(f"{len(unprocessed)} clips have no role-processing record. Complete automatic tracking before a full multi-view run; use --views scene for the full-scene control.")
    tr, va = rows[rows.fold != args.fold], rows[rows.fold == args.fold]
    if tr.empty or va.empty or not set(tr.source_id).isdisjoint(va.source_id):
        raise ValueError("Need nonempty source-disjoint train and validation sets")
    seed_all(args.seed)
    encoder, manifest = make_encoder(args.weights) if args.finetune else (None, json.loads((args.cache / "encoder_manifest.json").read_text()))
    model_config = {**MODEL_CONFIG, 'aggregation': getattr(args, 'aggregation', 'gru')}
    model = Predictor(encoder, model_config, finetune=args.finetune).to(args.device)
    if args.init_checkpoint:
        warm_start_temporal(model, args.init_checkpoint, rows, args, manifest)
    params = [{"params": model.temporal.parameters(), "lr": args.lr}]
    if encoder is not None:
        params.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(params, weight_decay=.01)
    config = {"model": model_config, "data": data.CONFIG, "encoder": manifest,
              "seed": args.seed, "fold": args.fold, "views": args.views, "finetune": args.finetune,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "encoder_lr": args.encoder_lr, "smoke_only": args.smoke,
              "initial_checkpoint_sha256": data.sha(args.init_checkpoint) if args.init_checkpoint else None,
              "max_train_batches": args.max_train_batches,
              "train_clips": tr["clip"].tolist(), "validation_clips": va["clip"].tolist(),
              "train_sources": sorted(tr.source_id.unique()), "validation_sources": sorted(va.source_id.unique()),
              "fold_csv_sha256": data.sha(args.cache / "public_folds.csv"),
              "input_provenance": provenance,
              "code_sha256": {"rgb_gru.py": data.sha(__file__), "rgb_gru_data.py": data.sha(data.__file__)},
              "device": args.device, "torch": str(torch.__version__),
              "interpretation": "Pipeline smoke test only; no generalization claim" if args.smoke else "Source-grouped validation; prior perception exposure is documented"}
    train_batches, val_batches = loader(tr, args, mode, True), loader(va, args, mode)
    history, best, start_epoch = [], -1., 0
    if resume:
        saved = restore_training(args.run / 'last.pt', model, optimizer, train_batches.generator, config)
        history, best, start_epoch = saved['history'], saved['best_score'], saved['epoch']
    else:
        data.dump(args.run / 'config.json', config)
    majority = np.array([values.index(tr[key].mode().iloc[0]) for key, values in data.TARGETS.items()])
    truth = np.array([[values.index(row[key]) for key, values in data.TARGETS.items()] for _, row in va.iterrows()])
    data.dump(args.run / "train_only_majority_reference.json", metric_report(truth, np.tile(majority, (len(va), 1))))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(train_batches):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            batch = to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = loss_for(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise ValueError("Non-finite gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, records = evaluate(model, val_batches, args.device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": metrics})
        data.dump(args.run / "history.json", history)
        if metrics["weighted_score"] > best:
            best = metrics["weighted_score"]
            atomic_save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                         "config": config, "epoch": epoch + 1, "metrics": metrics}, args.run / "best.pt")
            data.dump(args.run / "validation_predictions.json", records)
        atomic_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': config, 'epoch': epoch + 1, 'history': history, 'best_score': best,
                     'rng': capture_rng(train_batches.generator)}, args.run / 'last.pt')
        print(json.dumps({"epoch": epoch + 1, "train_loss": history[-1]["train_loss"],
                          "validation_score": metrics["weighted_score"], "smoke_only": args.smoke}), flush=True)
        if (getattr(args, 'max_seconds', None) and time.monotonic() - started >= args.max_seconds
                and epoch + 1 < args.epochs):
            data.dump(args.run / 'progress.json', {'state': 'checkpointed', 'epoch': epoch + 1,
                                                  'total_epochs': args.epochs})
            return False
    data.dump(args.run / "summary.json", {"best_validation_score": best, "smoke_only": args.smoke,
              "train_clips": len(tr), "validation_clips": len(va), "checkpoint_sha256": data.sha(args.run / "best.pt"),
              "teacher_score_not_comparable": 0.66944})
    if args.finetune:
        saved = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=True)
        atomic_save({'state_dict': {k.removeprefix('encoder.'): v for k, v in saved['state_dict'].items()
                                   if k.startswith('encoder.')},
                     'encoder': manifest, 'fold': args.fold, 'train_clips': config['train_clips'],
                     'validation_clips': config['validation_clips'], 'train_sources': config['train_sources'],
                     'validation_sources': config['validation_sources'],
                     'source_checkpoint_sha256': data.sha(args.run / 'best.pt'),
                     'selection': 'best fold-validation epoch; not independent of this validation set'},
                    args.run / 'backbone.pt')
    data.dump(args.run / 'progress.json', {'state': 'complete', 'epoch': args.epochs})
    return True


def prediction_table(records, expected):
    columns = ["clip", *data.TARGETS]
    table = pd.DataFrame(records)[columns]
    if table["clip"].duplicated().any() or set(table["clip"]) != set(expected):
        raise ValueError("Prediction coverage mismatch")
    for key, labels in data.TARGETS.items():
        if not table[key].isin(labels).all():
            raise ValueError(f"Invalid predicted labels: {key}")
    return table.set_index("clip").loc[list(expected)].reset_index()


def predict(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["data"] != data.CONFIG:
        raise ValueError("Checkpoint/preprocessing config mismatch")
    args.views = config["views"]
    mode = "images" if config["finetune"] else "features"
    if args.partition == "private":
        if config["smoke_only"]:
            raise ValueError("Smoke-test checkpoints cannot generate private submissions")
        rows = pd.read_csv(args.data_root / "private_metadata.csv")
        expected = pd.read_csv(args.data_root / "submission_template.csv")["clip"].tolist()
    else:
        all_rows = pd.read_csv(args.cache / "public_folds.csv")
        if data.sha(args.cache / "public_folds.csv") != config["fold_csv_sha256"]:
            raise ValueError("Changed validation fold manifest")
        expected = config["validation_clips"]
        rows = all_rows[all_rows["clip"].isin(expected)]
    provenance = validate_caches(rows, args.cache, mode)
    if args.partition == "validation":
        for clip, values in provenance.items():
            if values != config["input_provenance"][clip]:
                raise ValueError(f"Validation inputs changed since training: {clip}")
    encoder = None
    if config["finetune"]:
        encoder, manifest = make_encoder(args.weights)
    else:
        manifest = json.loads((args.cache / "encoder_manifest.json").read_text())
    if manifest != config["encoder"]:
        raise ValueError("Checkpoint/encoder mismatch")
    model = Predictor(encoder, config["model"], config["finetune"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics, records = evaluate(model.to(args.device), loader(rows, args, mode), args.device)
    table = prediction_table(records, expected)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    if args.predictions.exists():
        raise ValueError("Prediction output exists; choose a new path")
    table.to_csv(args.predictions, index=False)
    data.dump(args.predictions.with_suffix(".json"), {"checkpoint_sha256": data.sha(args.checkpoint),
              "smoke_only": config["smoke_only"], "metrics": metrics, "records": records})
    print(f"Wrote {len(table)} {args.partition} predictions; smoke_only={config['smoke_only']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["download", "encode", "train", "predict"])
    p.add_argument("--cache", type=Path, default=data.DEFAULT_OUT)
    p.add_argument("--data-root", type=Path, default=data.DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=data.ROOT / "models/efficientnet-b0")
    p.add_argument("--run", type=Path, default=data.ROOT / "outputs/rgb_gru/smoke_v1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=6643)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--views", choices=["scene", "players", "all"], default="all")
    p.add_argument("--finetune", action="store_true")
    p.add_argument('--aggregation', choices=['mean', 'gru'], default='gru')
    p.add_argument('--resume', action='store_true', help='Restore last.pt including optimizer and RNG state')
    p.add_argument('--max-seconds', type=float, help='Stop cleanly after a clip or epoch and preserve checkpoints')
    p.add_argument("--init-checkpoint", type=Path, help="Initialize GRU/heads from a matching frozen-encoder run")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--partition", choices=["validation", "private"], default="validation")
    p.add_argument("--predictions", type=Path, default=data.ROOT / "outputs/rgb_gru/validation.csv")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.max_train_batches is not None and args.max_train_batches < 1):
        p.error("Epochs, batch size and max train batches must be positive")
    if args.max_train_batches and not args.smoke:
        p.error("Partial-epoch training is reserved for --smoke")
    if args.init_checkpoint and (args.stage != "train" or not args.finetune):
        p.error("--init-checkpoint requires train --finetune")
    torch.set_num_threads(4)
    if args.stage == "download":
        _, manifest = make_encoder(args.weights, download=True)
        print(json.dumps(manifest, indent=2))
    elif args.stage == "encode":
        encode(args)
    elif args.stage == "train":
        train(args)
    elif args.stage == "predict":
        if args.checkpoint is None:
            p.error("predict requires --checkpoint")
        predict(args)


if __name__ == "__main__":
    main()
