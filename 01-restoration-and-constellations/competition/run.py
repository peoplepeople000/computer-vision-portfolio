"""Generate reproducible constellation competition predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from nocpt.classifier import CandidateClassifier
from nocpt.data import scene_rows, write_csv
from nocpt.patterns import load_patterns
from nocpt.pipeline import predict_scene
from nocpt.retrieval import retrieve_scene
from nocpt.deep_retrieval import retrieve_deep_scene
from nocpt.hybrid_retrieval import retrieve_hybrid_scene
from nocpt.union_retrieval import retrieve_union_scene
from nocpt.legacy_retrieval import load_legacy_scene


def main() -> None:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--highpass-always", action="store_true")
    parser.add_argument("--deep", action="store_true", help="Use the wider two-channel candidate search")
    parser.add_argument("--hybrid", action="store_true", help="Augment the short search with selected deep candidates")
    parser.add_argument("--union", action="store_true", help="Combine current and preserved legacy image candidates")
    parser.add_argument("--legacy", action="store_true", help="Use only preserved legacy image candidates")
    parser.add_argument("--fast", action="store_true", help="Use only the fast shortlist search")
    parser.add_argument("--model", type=Path, default=Path("models/candidate.json"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--prevalence-strength", type=float, default=None)
    parser.add_argument("--min-present-fraction", type=float, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    workers = args.workers if args.workers is not None else int(config.get("workers", 4))
    highpass_always = args.highpass_always or bool(config.get("highpass_always", False))
    requested_modes = sum((args.fast, args.deep, args.hybrid, args.union, args.legacy))
    if requested_modes > 1:
        raise ValueError("Select only one search mode")
    search_mode = ("legacy" if args.legacy else "union" if args.union else "fast" if args.fast else "deep" if args.deep else
                   "hybrid" if args.hybrid else config.get("search_mode", "fast"))
    if search_mode not in ("fast", "deep", "hybrid", "union", "legacy"):
        raise ValueError(f"Unsupported search mode: {search_mode}")
    prevalence_strength = (args.prevalence_strength if args.prevalence_strength is not None
                           else float(config.get("prevalence_strength", 0.0)))
    if not 0 <= prevalence_strength <= 1:
        raise ValueError("prevalence strength must be in [0, 1]")
    min_present_fraction = (args.min_present_fraction if args.min_present_fraction is not None
                            else float(config.get("min_present_fraction", 0.0)))
    if not 0 <= min_present_fraction <= 1:
        raise ValueError("minimum present fraction must be in [0, 1]")
    headers, rows = scene_rows(args.data_root, args.split)
    if args.scene:
        rows = [row for row in rows if row["Id"] == args.scene]
        if not rows:
            raise ValueError(f"Unknown scene: {args.scene}")
    output = args.output or Path("outputs") / (
        "submission.csv" if args.split == "validation" else "train_predictions.csv")
    partial_output = output.with_suffix(".partial.csv")
    diagnostics = output.parent / f"{output.stem}_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    model = CandidateClassifier.load(args.model)
    threshold_data = json.loads((args.model.parent / "threshold.json").read_text())
    threshold = threshold_data["probability_threshold"]
    target_present_fraction = threshold_data.get("training_present_fraction", 0.61)
    if config.get("probability_threshold") is not None:
        threshold = float(config["probability_threshold"])
    patterns = load_patterns(args.data_root)
    for row in rows:
        scene = row["Id"]
        if search_mode == "legacy":
            candidates = {number: [candidate] for number, candidate in
                          load_legacy_scene(args.data_root, args.split, scene).items()}
        elif search_mode == "union":
            candidates = retrieve_union_scene(args.data_root, args.split, scene,
                                              args.cache, workers=workers)
        elif search_mode == "hybrid":
            candidates = retrieve_hybrid_scene(args.data_root, args.split, scene,
                                               args.cache, workers=workers)
        elif search_mode == "deep":
            candidates = retrieve_deep_scene(args.data_root, args.split, scene, args.cache,
                                             workers=workers)
        else:
            candidates = retrieve_scene(args.data_root, args.split, scene, args.cache,
                                        workers=workers,
                                        highpass_always=highpass_always)
        (diagnostics / f"{scene}_candidates.json").write_text(json.dumps({
            "scene": scene,
            "patches": {f"patch_{number:02}": [c.serializable() for c in options]
                        for number, options in candidates.items()},
        }, indent=2))
        prediction = predict_scene(args.data_root, args.split, row,
                                   candidates, patterns, model, threshold,
                                   diagnostics / "results", target_present_fraction,
                                   prevalence_strength, min_present_fraction)
        row.update(prediction)
        write_csv(partial_output, headers, rows)
        print(f"Processed {scene} ({row['constellation']})", flush=True)
    os.replace(partial_output, output)
    manifest = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "data_root": str(args.data_root),
        "split": args.split,
        "scene_count": len(rows),
        "search_mode": search_mode,
        "workers": workers,
        "probability_threshold": threshold,
        "min_present_fraction": min_present_fraction,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "config_sha256": (hashlib.sha256(args.config.read_bytes()).hexdigest()
                          if args.config.exists() else None),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output": str(output),
    }
    if search_mode in ("union", "legacy"):
        polar_directory = Path("reference/legacy_dense") / args.split
        manifest["polar_cache_sha256"] = {
            row["Id"]: hashlib.sha256((polar_directory /
                (f"{row['Id']}_dense_phase_predictions.csv" if args.split == "train"
                 else f"{row['Id']}_locations.csv")).read_bytes()).hexdigest()
            for row in rows
        }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
