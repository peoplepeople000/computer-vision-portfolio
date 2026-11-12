"""Untouched SAM encoder cache and quality-selected multimask refinement.

Ownership and original shape guards remain identical to the prior fixed-center
pipeline. No target annotations are used by this inference module.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

from cellmatch.data import parse_instances, read_rows
from cellmatch.direct_center_ex import sha
from cellmatch.rle import encode
from cellmatch.sam_instance_output_adaptation import MODEL, decode, field

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "colab/sam_lora_v1/input"
BASELINES = ("research328", "production385")
HELD_SUBJECT = "subject_b2ba5e"
MAX_SECONDS = 600.0
MAX_DRIVER_BYTES = 6 * 1024**3


def _check_encoder_frozen(sam):
    trainable = [name for name, p in sam.named_parameters() if p.requires_grad and name.startswith("image_encoder.")]
    if trainable:
        raise RuntimeError(f"Image encoder must remain frozen: {trainable[:4]}")
    if sam.training or sam.image_encoder.training:
        raise RuntimeError("SAM and its image encoder must be in eval mode")


def _gray_crop(gray: np.ndarray, x0: int, y0: int) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected normalized uint8 grayscale image")
    h, w = gray.shape
    tile = np.zeros((256, 256), np.uint8)
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + 256, w), min(y0 + 256, h)
    if x2 > x1 and y2 > y1:
        tile[y1-y0:y2-y0, x1-x0:x2-x0] = gray[y1:y2, x1:x2]
    return tile


def _component(mask: np.ndarray, point: np.ndarray) -> np.ndarray:
    labels, _ = label(mask)
    ys, xs = np.nonzero(labels)
    if not len(xs):
        return np.zeros_like(mask)
    d2 = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    nearest = int(d2.argmin())
    if d2[nearest] > 16:
        return np.zeros_like(mask)
    return labels == labels[ys[nearest], xs[nearest]]


def _crop_pixels(local: np.ndarray, x0: int, y0: int, shape: tuple[int, int]):
    ys, xs = np.nonzero(local)
    ys, xs = ys + y0, xs + x0
    valid = (ys >= 0) & (xs >= 0) & (ys < shape[0]) & (xs < shape[1])
    return ys[valid], xs[valid]


def _center_origin(center: np.ndarray) -> tuple[int, int]:
    x0, y0 = (np.floor(np.asarray(center, dtype=np.float64) / 128).astype(int) * 128 - 64).tolist()
    return int(x0), int(y0)


def _matched_rows():
    result = {}
    for branch in BASELINES:
        rows = read_rows(DATA / "baselines" / f"{branch}.csv")
        held = {sid: row for sid, row in rows.items() if sid.split("__", 1)[0] == HELD_SUBJECT}
        if len(held) != 13:
            raise ValueError(f"Expected 13 held rows for {branch}, got {len(held)}")
        result[branch] = held
    return result


def prepare_fields(sam, out: Path) -> dict:
    """Cache one frozen image-encoder field per old fixed-center crop.

    Covers all selected EX endpoints in the 13 held rows of both fixed local
    baseline branches. The returned mapping is keyed by ``(sample_id, x0, y0)``.
    """
    _check_encoder_frozen(sam)
    out = Path(out).expanduser().resolve()
    field_dir = out / "held_fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    index_path = out / "held_fields_index.json"
    if index_path.exists():
        raise FileExistsError(f"Refusing to replace existing field index: {index_path}")

    branch_rows = _matched_rows()
    groups: dict[tuple[str, int, int], dict] = {}
    input_hashes = {}
    images = {}
    for branch, rows in branch_rows.items():
        csv_path = DATA / "baselines" / f"{branch}.csv"
        input_hashes[str(csv_path)] = sha(csv_path)
        for sid, row in rows.items():
            image_path = DATA / "held_ex" / f"{sid}.npz"
            if sid not in images:
                input_hashes[str(image_path)] = sha(image_path)
                with np.load(image_path) as saved:
                    gray = saved["gray"].copy()
                if gray.ndim != 2 or gray.dtype != np.uint8:
                    raise ValueError(f"Bad held image payload: {image_path}")
                images[sid] = gray
            shape = images[sid].shape
            bank = parse_instances(row, "exvivo", shape)
            keys = sorted({ex for _iv, ex in row["match_pairs"]})
            for key in keys:
                if key not in bank or bank[key].area <= 0:
                    raise ValueError(f"Invalid selected EX endpoint {sid}/{key}")
                x0, y0 = _center_origin(bank[key].coordinates().mean(0))
                group_key = (sid, x0, y0)
                group = groups.setdefault(group_key, {"keys": {b: set() for b in BASELINES}, "shape": list(shape)})
                if group["shape"] != list(shape):
                    raise ValueError(f"Conflicting image shape for {sid}")
                group["keys"][branch].add(key)

    started = time.monotonic()
    fields = {}
    records = []
    for (sid, x0, y0), group in sorted(groups.items()):
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError(f"Encoder field preparation exceeded {MAX_SECONDS:.0f}s")
        tensor = field(sam, _gray_crop(images[sid], x0, y0))
        if tensor.shape != (1, 256, 64, 64) or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid encoder field for {(sid, x0, y0)}")
        if torch.backends.mps.is_available() and next(sam.parameters()).device.type == "mps":
            torch.mps.synchronize()
            driver = int(torch.mps.driver_allocated_memory())
            if driver > MAX_DRIVER_BYTES:
                raise MemoryError(f"MPS driver allocation {driver} exceeds {MAX_DRIVER_BYTES}")
        else:
            driver = None
        field_np = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        path = (field_dir / f"{sid}__x{x0}_y{y0}.npz").resolve()
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing cached field: {path}")
        np.savez_compressed(path, field=field_np)
        digest = sha(path)
        item = {"field_path": str(path), "field_sha256": digest}
        fields[(sid, x0, y0)] = item
        records.append({"sample_id": sid, "origin_xy": [x0, y0], "field_path": str(path),
                        "field_sha256": digest, "field_shape": list(field_np.shape),
                        "field_dtype": str(field_np.dtype), "selected_ex_keys": {
                            b: sorted(group["keys"][b]) for b in BASELINES}})
        del tensor, field_np
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError(f"Encoder field preparation exceeded {MAX_SECONDS:.0f}s")
    index = {"status": "complete", "held_subject": HELD_SUBJECT, "regions_per_branch": 13,
             "groups": len(records), "crop_shape": [256, 256], "field_shape": [1, 256, 64, 64],
             "source_sha256": sha(Path(__file__)),
             "continuation_sha256": sha(ROOT / "src/cellmatch/sam_instance_output_adaptation.py"),
             "base_model_sha256": sha(MODEL), "encoder_state": "untouched_official_backbone_no_adapters",
             "inputs": input_hashes, "fields": records,
             "elapsed_seconds": time.monotonic() - started}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    if len(fields) != len(groups):
        raise AssertionError("Not every unique held crop produced a field")
    return fields


def _load_field(entry, device):
    path = Path(entry["field_path"])
    if not path.is_absolute() or sha(path) != entry["field_sha256"]:
        raise RuntimeError(f"Cached encoder field checksum/path mismatch: {path}")
    with np.load(path) as saved:
        if set(saved.files) != {"field"}:
            raise ValueError(f"Unexpected arrays in cached field: {path}")
        value = saved["field"]
    if value.shape != (1, 256, 64, 64) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid cached field payload: {path}")
    return torch.as_tensor(value, device=device)


def predict_rows(sam, rows: dict, fields: dict) -> list:
    """Apply quality-selected multimasks with fixed-center sequential guards.

    ``rows`` maps held sample IDs to original baseline rows. All link IDs and
    endpoint sets remain fixed; only masks for currently paired EX keys may be
    refined. Decoding is chunked to at most eight prompts per tile.
    """
    _check_encoder_frozen(sam)
    device = next(sam.parameters()).device
    returned = []
    for sid, source_row in sorted(rows.items()):
        if sid.split("__", 1)[0] != HELD_SUBJECT:
            raise ValueError(f"predict_rows accepts held rows only: {sid}")
        image_path = DATA / "held_ex" / f"{sid}.npz"
        with np.load(image_path) as saved:
            gray = saved["gray"].copy()
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError(f"Bad held image payload: {image_path}")
        row = copy.deepcopy(source_row)
        original_pairs = copy.deepcopy(row["match_pairs"])
        bank = parse_instances(row, "exvivo", gray.shape)
        keys = sorted({ex for _iv, ex in original_pairs})
        owners = {k: i + 1 for i, k in enumerate(sorted(bank))}
        occupied = np.zeros(gray.shape, np.int32)
        for key, mask in bank.items():
            xy = mask.coordinates()
            occupied[xy[:, 1], xy[:, 0]] = owners[key]
        original_occupied = occupied.copy()
        groups = {}
        for key in keys:
            if key not in bank or bank[key].area <= 0:
                raise ValueError(f"Invalid paired EX endpoint {sid}/{key}")
            center = bank[key].coordinates().mean(0)
            x0, y0 = _center_origin(center)
            groups.setdefault((x0, y0), []).append((key, center - [x0, y0]))

        changed = 0
        for (x0, y0), items in sorted(groups.items()):
            try:
                entry = fields[(sid, x0, y0)]
            except KeyError as error:
                raise KeyError(f"Missing shared encoder field for {(sid, x0, y0)}") from error
            feature = _load_field(entry, device)
            for begin in range(0, len(items), 8):
                chunk = items[begin:begin + 8]
                points = np.asarray([point for _key, point in chunk], dtype=np.float32)
                if points.shape != (len(chunk), 2) or not np.isfinite(points).all():
                    raise ValueError(f"Invalid decoder points in {sid} at {(x0, y0)}")
                with torch.no_grad():
                    logits, scores = decode(sam, feature, points)
                    assert logits.shape == (len(chunk), 3, 256, 256) and scores.shape == (len(chunk), 3)
                    assert torch.isfinite(scores).all()
                    choice = scores.argmax(dim=1)
                    selected = logits[torch.arange(len(chunk), device=device), choice]
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite decoder logits for {sid} at {(x0, y0)}")
                if torch.backends.mps.is_available() and device.type == "mps":
                    torch.mps.synchronize()
                    driver = int(torch.mps.driver_allocated_memory())
                    if driver > MAX_DRIVER_BYTES:
                        raise MemoryError(f"MPS driver allocation {driver} exceeds {MAX_DRIVER_BYTES}")
                masks = (selected.detach().cpu().numpy() > 0)
                if masks.shape != (len(chunk), 256, 256):
                    raise RuntimeError(f"Unexpected decoded mask shape {masks.shape}")
                for (key, _point), mask in zip(chunk, masks):
                    local = _component(mask, _point)
                    ys, xs = _crop_pixels(local, x0, y0, gray.shape)
                    owner_ids = occupied[ys, xs]
                    original_ids = original_occupied[ys, xs]
                    keep = ((owner_ids == 0) | (owner_ids == owners[key])) & ((original_ids == 0) | (original_ids == owners[key]))
                    ys, xs = ys[keep], xs[keep]
                    prior = bank[key]
                    area = len(xs)
                    if not .5 * prior.area <= area <= 1.7 * prior.area:
                        continue
                    candidate = np.zeros(gray.shape, dtype=bool)
                    candidate[ys, xs] = True
                    old = prior.decode()
                    intersection = int((candidate & old).sum())
                    union = area + prior.area - intersection
                    if not union or intersection / union <= .4:
                        continue
                    if np.array_equal(candidate, old):
                        continue
                    # Match the original sequential ownership update: release this
                    # cell's old pixels, then reserve the accepted replacement.
                    occupied[old & (occupied == owners[key])] = 0
                    occupied[ys, xs] = owners[key]
                    row["exvivo_instances"][key] = encode(candidate)
                    changed += 1
            del feature
        if row["match_pairs"] != original_pairs or row["invivo_instances"] != source_row["invivo_instances"]:
            raise AssertionError(f"Links or IV masks changed while refining {sid}")
        returned.append(row)
    return returned
