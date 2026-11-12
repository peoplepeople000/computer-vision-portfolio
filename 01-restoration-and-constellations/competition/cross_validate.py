"""Evaluate the complete pipeline by leaving out one labeled scene at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nocpt.classifier import CandidateClassifier, FEATURE_NAMES, UNION_FEATURE_NAMES, feature_vector
from nocpt.data import parse_cell, read_csv, write_csv
from nocpt.deep_retrieval import retrieve_deep_scene
from nocpt.patterns import load_patterns
from nocpt.pipeline import predict_scene
from nocpt.retrieval import retrieve_scene
from nocpt.union_retrieval import retrieve_union_scene
from nocpt.legacy_retrieval import load_legacy_scene
from nocpt.scoring import score_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cv_predictions.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-present-fraction", type=float, default=0.55)
    parser.add_argument("--union", action="store_true",
                        help="Use both current and preserved legacy image candidates")
    parser.add_argument("--legacy", action="store_true",
                        help="Use only preserved legacy image candidates")
    args = parser.parse_args()
    headers, truth = read_csv(args.data_root / "train_ground_truth.csv")
    patterns = load_patterns(args.data_root)
    training_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in truth:
        scene = row["Id"]
        image = cv2.imread(str(args.data_root / "train" / scene /
                               f"{scene}_image.png"), cv2.IMREAD_GRAYSCALE)
        candidates = ({number: [candidate] for number, candidate in
                       load_legacy_scene(args.data_root, "train", scene).items()}
                      if args.legacy else
                      retrieve_union_scene(args.data_root, "train", scene, args.cache,
                                           workers=args.workers) if args.union else
                      retrieve_scene(args.data_root, "train", scene, args.cache,
                                     workers=args.workers))
        features, labels = [], []
        for number in range(1, int(row["n_patches"]) + 1):
            patch = cv2.imread(str(args.data_root / "train" / scene / "patches" /
                                   f"patch_{number:02}.png"), cv2.IMREAD_GRAYSCALE)
            expected = parse_cell(row[f"patch_{number:02}"])
            for rank, candidate in enumerate(candidates[number]):
                vector = feature_vector(image, patch, candidate,
                                        0 if candidate.channel == "legacy" else rank)
                if args.union:
                    vector = np.append(vector, float(candidate.channel == "legacy"))
                features.append(vector)
                labels.append(int(expected is not None and np.hypot(
                    candidate.x-expected[0], candidate.y-expected[1]) <= 12))
        training_groups[scene] = np.stack(features), np.asarray(labels)
    predictions = []
    diagnostics = args.output.parent / f"{args.output.stem}_diagnostics"
    for row in truth:
        scene = row["Id"]
        x = np.concatenate([features for name, (features, _) in training_groups.items()
                            if name != scene])
        y = np.concatenate([labels for name, (_, labels) in training_groups.items()
                            if name != scene])
        model = CandidateClassifier.fit(x, y,
                                        UNION_FEATURE_NAMES if args.union else FEATURE_NAMES)
        candidates = ({number: [candidate] for number, candidate in
                       load_legacy_scene(args.data_root, "train", scene).items()}
                      if args.legacy else
                      retrieve_union_scene(args.data_root, "train", scene, args.cache,
                                           workers=args.workers) if args.union else
                      retrieve_deep_scene(args.data_root, "train", scene, args.cache,
                                          workers=args.workers))
        prediction = predict_scene(args.data_root, "train", row, candidates,
                                   patterns, model, args.threshold,
                                   diagnostics, min_present_fraction=args.min_present_fraction)
        predictions.append(prediction)
        result = score_scene(row, prediction)
        print(f"{scene}: score={result['score']:.4f} name={prediction['constellation']}",
              flush=True)
    write_csv(args.output, headers, predictions)
    mean = float(np.mean([score_scene(a, b)["score"]
                          for a, b in zip(truth, predictions)]))
    print(f"Full leave-one-scene-out mean: {mean:.4f}")
    (args.output.parent / "cv_summary.json").write_text(json.dumps({
        "full_leave_one_scene_out_mean": mean,
        "threshold": args.threshold,
        "min_present_fraction": args.min_present_fraction,
        "scene_scores": {a["Id"]: score_scene(a, b)
                         for a, b in zip(truth, predictions)},
    }, indent=2))


if __name__ == "__main__":
    main()
