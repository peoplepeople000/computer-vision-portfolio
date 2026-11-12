"""Recompute one image-only contrast-phase candidate and audit its saved cache.

This is a provenance check for the preserved retrieval bank, not an inference
path. It reads the scene image, query patch, and optional image-derived cache
row; it never reads training labels, a submission, or a constellation name.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from generate_polar_cache import load_polar_module, read_gray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--patch", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("reference/legacy_dense"))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    folder = args.data_root / args.split / args.scene
    image = read_gray(folder / f"{args.scene}_image.png")
    patch = read_gray(folder / "patches" / f"patch_{args.patch:02d}.png")
    module = load_polar_module(
        Path("reference/legacy_dense/q5_experimental_functions.py"))
    started = time.monotonic()
    index = module.build_contrast_index(image, stride=args.stride,
                                        scales=(0.8, 1.0, 1.2))
    index["features"] = []
    module.add_phase_index(index)
    centers, _ = module.contrast_phase_candidates(
        index, patch, top_k=args.top_k)
    candidate = {key: float(value) for key, value in
                 module.dense_contrast_match(index, patch, centers).items()}

    filename = (f"{args.scene}_dense_phase_predictions.csv"
                if args.split == "train" else f"{args.scene}_locations.csv")
    cache_path = args.cache_root / args.split / filename
    with cache_path.open(newline="") as handle:
        saved = next((row for row in csv.DictReader(handle)
                      if int(row["patch_id"]) == args.patch), None)
    if saved is None:
        raise ValueError(f"Patch {args.patch} missing from {cache_path}")
    compared = {key: float(saved[key]) for key in candidate}
    report = {
        "split": args.split,
        "scene": args.scene,
        "patch": args.patch,
        "method": "contrast_phase",
        "stride": args.stride,
        "top_k": args.top_k,
        "recomputed": candidate,
        "preserved": compared,
        "preserved_new_candidate_selected": saved["new_candidate_selected"],
        "center_difference_px": float(np.hypot(
            candidate["x"] - compared["x"],
            candidate["y"] - compared["y"])),
        "score_difference": candidate["score"] - compared["score"],
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
