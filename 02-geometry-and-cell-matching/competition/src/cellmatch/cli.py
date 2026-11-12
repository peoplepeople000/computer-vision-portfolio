"""Inspect data, generate predictions, evaluate, and validate submissions."""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .data import (read_rows, read_image, image_path, image_shape, parse_instances,
                   split_id, write_rows, validate_submission)
from .rle import Mask, labels_to_instances
from .metrics import segmentation_score, pair_score, aggregate
from .matching import match_cells
from .segmentation import segment


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect_data(args):
    root = Path(args.data)
    train = read_rows(root / "training/train_ground_truth.csv")
    test = read_rows(root / "sample_submission.csv")
    subjects = sorted({split_id(sid)[0] for sid in train})
    test_subjects = sorted({split_id(sid)[0] for sid in test})
    if set(subjects) & set(test_subjects):
        raise ValueError("Training/test subjects overlap")
    if (len(train), len(test), len(subjects), len(test_subjects)) != (47, 29, 3, 2):
        raise ValueError("Unexpected official dataset counts (expected 47/29 regions, 3/2 subjects)")
    records, hashes = [], {}
    for split, rows in (("training", train), ("hidden_test", test)):
        for sid, row in rows.items():
            record = {"sample_id": sid, "subject": split_id(sid)[0], "split": split,
                      "verified_pairs": len(row["match_pairs"]) if split == "training" else None}
            for modality in ("invivo", "exvivo"):
                path = image_path(root, split, sid, modality)
                image = read_image(path)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                hashes.setdefault((modality, image.shape, str(image.dtype), digest), []).append(sid)
                info = {"shape": list(image.shape), "dtype": str(image.dtype), "sha256_pixels": digest,
                        "percentiles": np.percentile(image, [0, 1, 50, 99, 99.9, 100]).tolist()}
                if split == "training":
                    masks = parse_instances(row, modality, image.shape)
                    areas = [m.area for m in masks.values()]
                    info.update(instances=len(masks), empty_masks=sum(a == 0 for a in areas),
                                area_quantiles=np.percentile(areas, [0, 25, 50, 75, 100]).tolist() if areas else [])
                record[modality] = info
            records.append(record)
    folds = [{"name": f"holdout_{subject}", "held_out_subject": subject,
              "train_ids": [sid for sid in train if split_id(sid)[0] != subject],
              "validation_ids": [sid for sid in train if split_id(sid)[0] == subject]} for subject in subjects]
    duplicate_groups = [{"modality": key[0], "sample_ids": ids} for key, ids in hashes.items() if len(ids) > 1]
    report = {"training_regions": len(train), "test_regions": len(test), "training_subjects": subjects,
              "test_subjects": test_subjects, "duplicate_image_groups": duplicate_groups,
              "training_cells": {m: sum(r[m]["instances"] for r in records if r["split"] == "training") for m in ("invivo", "exvivo")},
              "verified_pairs": sum(len(row["match_pairs"]) for row in train.values()), "regions": records}
    output = Path(args.output)
    save_json(report, output / "dataset_audit.json")
    save_json(folds, output / "subject_folds.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("regions", "duplicate_image_groups")}, indent=2))
    print(f"Duplicate image groups: {len(duplicate_groups)}. Reports saved to {output}")


def predict(args):
    root = Path(args.data)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    config = json.loads(Path(args.config).read_text())
    source = root / ("training/train_ground_truth.csv" if args.split == "training" else "sample_submission.csv")
    rows = read_rows(source)
    selected = [(sid, row) for sid, row in rows.items() if args.subject is None or split_id(sid)[0] == args.subject]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No selected regions")
    if args.oracle_masks and args.split != "training":
        raise ValueError("Oracle masks are a training-data diagnostic only")
    predictions, diagnostics = [], []
    started = time.perf_counter()
    for number, (sid, truth) in enumerate(selected, 1):
        tick = time.perf_counter()
        prediction = {"sample_id": sid}
        masks = {}
        for modality, prefix in (("invivo", "IVP"), ("exvivo", "EXP")):
            image = read_image(image_path(root, args.split, sid, modality))
            instances = truth[f"{modality}_instances"] if args.oracle_masks else labels_to_instances(segment(image, config[modality]), prefix)
            prediction[f"{modality}_instances"] = instances
            masks[modality] = {key: Mask.parse(rle, image.shape) for key, rle in instances.items()}
        # Never pass verified correspondences to the registration algorithm.
        pairs, diagnostic = match_cells(masks["invivo"], masks["exvivo"], config["matching"], config["seed"])
        prediction["match_pairs"] = pairs
        predictions.append(prediction)
        diagnostic.update(sample_id=sid, seconds=time.perf_counter()-tick,
                          invivo_cells=len(masks["invivo"]), exvivo_cells=len(masks["exvivo"]))
        diagnostics.append(diagnostic)
        print(f"[{number}/{len(selected)}] {sid}: {len(masks['invivo'])}/{len(masks['exvivo'])} cells, {len(pairs)} pairs, {diagnostic['seconds']:.1f}s", flush=True)
    write_rows(predictions, args.output)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(root.resolve()),
                "split": args.split, "oracle_masks": args.oracle_masks, "subject": args.subject,
                "config": config, "seconds": time.perf_counter()-started,
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-image", "tifffile")},
                "source_sha256": source_hashes,
                "prediction_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(), "regions": diagnostics}
    save_json(manifest, str(args.output) + ".manifest.json")
    print(f"Saved {args.output}")


def evaluate(args):
    root = Path(args.data)
    truth = read_rows(root / "training/train_ground_truth.csv")
    predictions = read_rows(args.predictions)
    expected = {sid for sid in truth if args.subject is None or split_id(sid)[0] == args.subject}
    if not expected or set(predictions) != expected:
        raise ValueError("Prediction IDs must exactly match all training regions or the requested holdout subject")
    results = []
    for sid, prediction in predictions.items():
        row = {"sample_id": sid, "subject": split_id(sid)[0]}
        for modality in ("invivo", "exvivo"):
            shape = image_shape(image_path(root, "training", sid, modality))
            row[modality] = segmentation_score(parse_instances(prediction, modality, shape), parse_instances(truth[sid], modality, shape))
        row["matching"] = pair_score(prediction["match_pairs"], truth[sid]["match_pairs"],
                                     row["invivo"]["pred_to_gt"], row["exvivo"]["pred_to_gt"])
        for modality in ("invivo", "exvivo"):
            row[modality].pop("pred_to_gt")
        results.append(row)
    report = {"metric_source": "Local implementation of published specification; official evaluator unavailable",
              "summary": aggregate(results),
              "by_subject": {s: aggregate([r for r in results if r["subject"] == s]) for s in sorted({r["subject"] for r in results})},
              "regions": results}
    save_json(report, args.output)
    print(json.dumps(report["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Audit official data and create three subject holdout folds")
    inspect.add_argument("--data", required=True)
    inspect.add_argument("--output", default="outputs/audit")
    inspect.set_defaults(func=inspect_data)
    prediction = commands.add_parser("predict", help="Run the untuned classical baseline")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--config", default="configs/baseline.json")
    prediction.add_argument("--split", choices=("training", "hidden_test"), required=True)
    prediction.add_argument("--subject")
    prediction.add_argument("--limit", type=int)
    prediction.add_argument("--oracle-masks", action="store_true")
    prediction.add_argument("--output", required=True)
    prediction.set_defaults(func=predict)
    evaluation = commands.add_parser("evaluate", help="Score training predictions without silently ignoring missing regions")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--subject")
    evaluation.add_argument("--output", default="outputs/evaluation.json")
    evaluation.set_defaults(func=evaluate)
    validation = commands.add_parser("validate", help="Check all 29 test rows, masks, and pair IDs")
    validation.add_argument("--data", required=True)
    validation.add_argument("--predictions", required=True)
    validation.set_defaults(func=lambda args: print(json.dumps(validate_submission(args.data, args.predictions), indent=2)))
    visual = commands.add_parser("preview", help="Render training ground truth and predicted contours")
    visual.add_argument("--data", required=True)
    visual.add_argument("--predictions", required=True)
    visual.add_argument("--sample-id", required=True)
    visual.add_argument("--output", required=True)
    def render(args):
        from .visualization import preview
        preview(args.data, args.predictions, args.sample_id, args.output)
        print(f"Saved {args.output}")
    visual.set_defaults(func=render)
    geometry = commands.add_parser("geometry-diagnostic", help="Fit known TRAINING correspondences to diagnose deformation")
    geometry.add_argument("--data", required=True)
    geometry.add_argument("--output", default="outputs/audit/geometry_diagnostic.json")
    def diagnose(args):
        from .diagnostics import geometry_diagnostic
        save_json(geometry_diagnostic(args.data), args.output)
        print(f"Saved {args.output}")
    geometry.set_defaults(func=diagnose)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
