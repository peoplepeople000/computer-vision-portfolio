"""Versioned, source-grouped RGB inputs. No position clicks enter model inputs.

prepare: split all eligible public clips; cache: export existing automatic tracks.
Tracks may be absent: full-scene RGB remains available, role masks stay false.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/kaggle"
DEFAULT_OUT = ROOT / "data/processed/rgb_gru_v1"
DEFAULT_TRACKS = ROOT / "outputs/fulltrack_pose_pilot/tracks"
TARGETS = {
    "kick_direction": ["left", "middle", "right"],
    "goalkeeper_direction": ["left", "stands in the middle", "right"],
    "goal_outcome": ["goal", "no_goal"],
}
WEIGHTS = [0.60, 0.25, 0.15]
CONFIG = {"version": 1, "seed": 6643, "folds": 5, "frames": 16,
          "tail_seconds": 0.6, "size": 224, "crop_margin": 0.20,
          "views": ["scene", "kicker", "keeper"],
          "sampling": "8 whole-clip plus 8 final-0.6s timestamps; unique chronological frames",
          "resize": "aspect-preserving letterbox; ImageNet mean fill; no horizontal flip",
          "metadata": ["preferred_foot"],
          "excluded": {"0526_public.mp4": "Previously flagged possible contact boundary; pending review"}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Changed protocol/input: use a new output directory: {path}")
    else:
        dump(path, value)


def prepare(data=DEFAULT_DATA, out=DEFAULT_OUT):
    data, out = Path(data), Path(out)
    source = data / "train_ground_truth.csv"
    public = pd.read_csv(source)
    if public["clip"].duplicated().any():
        raise ValueError("Duplicate public clips")
    for key, labels in TARGETS.items():
        if not public[key].isin(labels).all():
            raise ValueError(f"Invalid labels: {key}")
    public = public[~public["clip"].isin(CONFIG["excluded"])].copy().reset_index(drop=True)
    public["fold"] = -1
    splitter = StratifiedGroupKFold(CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    for fold, (_, validation) in enumerate(splitter.split(public, public.kick_direction, public.source_id)):
        public.loc[validation, "fold"] = fold
    if public.groupby("source_id")["fold"].nunique().max() != 1:
        raise AssertionError("Source leakage")
    path = out / "public_folds.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), public)
    else:
        out.mkdir(parents=True, exist_ok=True)
        public.to_csv(path, index=False)
    counts = {}
    for fold, rows in public.groupby("fold"):
        counts[str(fold)] = {"clips": len(rows), "sources": rows.source_id.nunique(),
                             "labels": {k: rows[k].value_counts().to_dict() for k in TARGETS}}
    immutable_json(out / "split_protocol.json", {
        "config": CONFIG, "public_csv_sha256": sha(source), "fold_csv_sha256": sha(path),
        "fold_counts": counts,
        "teacher_reference": {"architecture": "YOLO + GRU (user-reported; implementation unavailable)",
                              "kaggle_public_score": 0.66944,
                              "comparable_to_local_validation": False},
        "scope": "Source-disjoint classification folds; perception has previously inspected some sources. Not an untouched benchmark."})
    return public


def select_indices(times):
    times = np.asarray(times, dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Expected finite, strictly increasing presentation timestamps")
    half = CONFIG["frames"] // 2
    targets = np.r_[np.linspace(times[0], times[-1], half),
                    np.linspace(max(times[0], times[-1] - CONFIG["tail_seconds"]), times[-1], half)]
    return np.unique([np.abs(times - t).argmin() for t in targets])


def video_times(path):
    cap = cv2.VideoCapture(str(path))
    times = []
    try:
        while cap.grab():
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        cap.release()
    select_indices(times)  # Also rejects an empty, unreadable, or non-monotone clip.
    return np.asarray(times, np.float64)


def letterbox(rgb):
    size = CONFIG["size"]
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = np.round(np.array([.485, .456, .406]) * 255).astype(np.uint8)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def crop_view(rgb, box):
    if box is None or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    dx, dy = (x2 - x1) * CONFIG["crop_margin"], (y2 - y1) * CONFIG["crop_margin"]
    h, w = rgb.shape[:2]
    x1, x2 = np.clip([np.floor(x1 - dx), np.ceil(x2 + dx)], 0, w).astype(int)
    y1, y2 = np.clip([np.floor(y1 - dy), np.ceil(y2 + dy)], 0, h).astype(int)
    return letterbox(rgb[y1:y2, x1:x2]) if x2 > x1 and y2 > y1 else None


def cache_clip(clip, video, track_path, out):
    track_path, out = Path(track_path), Path(out)
    track = json.loads(track_path.read_text()) if track_path.exists() else None
    provenance = {"config": CONFIG, "clip": clip, "video_sha256": sha(video),
                  "track_sha256": sha(track_path) if track else None}
    target = out / "crops" / (clip + ".npz")
    meta = target.with_suffix(".json")
    if target.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved["inputs"] != provenance or saved["cache_sha256"] != sha(target):
            raise ValueError(f"Stale or corrupted crop cache: {clip}")
        return saved
    times = video_times(video)
    indices = select_indices(times)
    if track and (track["clip"] != clip or track["frame_count"] != len(times)):
        raise ValueError(f"Track/video mismatch: {clip}")
    records = {f["frame_index"]: f for f in track["frames"]} if track else {}
    images, masks = [], []
    cap = cv2.VideoCapture(str(video))
    selected = set(indices.tolist())
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in selected:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                views = np.zeros((3, CONFIG["size"], CONFIG["size"], 3), np.uint8)
                mask = np.array([True, False, False])
                views[0] = letterbox(rgb)
                record = records.get(i)
                if record:
                    if not np.isclose(record["timestamp"], times[i], atol=1e-4):
                        raise ValueError(f"Timestamp alignment mismatch: {clip}/{i}")
                    for ri, role in enumerate(["kicker", "keeper"], 1):
                        crop = crop_view(rgb, record["roles"][role]["bbox"])
                        if crop is not None:
                            views[ri], mask[ri] = crop, True
                images.append(views)
                masks.append(mask)
            i += 1
    finally:
        cap.release()
    if len(images) != len(indices):
        raise ValueError(f"Incomplete decoding: {clip}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, images=np.stack(images), view_mask=np.stack(masks),
                        timestamps=times[indices], frame_indices=indices)
    result = {"inputs": provenance, "cache_sha256": sha(target), "samples": len(indices),
              "view_coverage": np.mean(masks, axis=0).tolist(),
              "roles_available": track is not None}
    dump(meta, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cache"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--scope", choices=["tracked", "public", "private"], default="tracked")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        rows = prepare(args.data_root, args.out)
        print(f"Prepared {len(rows)} clips / {rows.source_id.nunique()} sources / 5 folds", flush=True)
        return
    if args.scope == "private":
        rows = pd.read_csv(args.data_root / "private_metadata.csv")
        partition = "val"
    else:
        rows = pd.read_csv(args.out / "public_folds.csv")
        partition = "train"
        if args.scope == "tracked":
            rows = rows[rows["clip"].map(lambda c: (args.tracks / (c + ".json")).exists())]
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        raise ValueError("No clips selected")
    report = []
    for clip in rows["clip"]:
        result = cache_clip(clip, args.data_root / partition / clip, args.tracks / (clip + ".json"), args.out)
        report.append({"clip": clip, "samples": result["samples"], "view_coverage": result["view_coverage"]})
        print(f"Cached {clip}: {result['samples']} times; coverage {result['view_coverage']}", flush=True)
    dump(args.out / f"cache_report_{args.scope}.json", report)


if __name__ == "__main__":
    main()
