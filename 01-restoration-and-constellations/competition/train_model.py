"""Fit the candidate verifier on labelled scenes and report leave-one-scene-out scores."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from nocpt.classifier import CandidateClassifier, FEATURE_NAMES, UNION_FEATURE_NAMES, feature_vector
from nocpt.data import read_csv, format_cell
from nocpt.matching import Candidate
from nocpt.scoring import score_scene
from nocpt.deep_retrieval import retrieve_deep_scene
from nocpt.hybrid_retrieval import retrieve_hybrid_scene
from nocpt.union_retrieval import retrieve_union_scene
from nocpt.legacy_retrieval import load_legacy_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", type=Path, default=Path("models/candidate.json"))
    parser.add_argument("--deep", action="store_true", help="Train with the wider candidate search")
    parser.add_argument("--hybrid", action="store_true", help="Train with the combined candidate search")
    parser.add_argument("--union", action="store_true", help="Train with current and legacy image candidates")
    parser.add_argument("--legacy", action="store_true", help="Train with preserved legacy image candidates")
    args = parser.parse_args()
    _, truth = read_csv(args.data_root / "train_ground_truth.csv")
    scene_data = {}
    for row in truth:
        scene = row["Id"]
        image = cv2.imread(str(args.data_root / "train" / scene / f"{scene}_image.png"), 0)
        if sum((args.deep, args.hybrid, args.union, args.legacy)) > 1:
            raise ValueError("Select one candidate search mode")
        if args.legacy:
            retrieved = {number: [candidate] for number, candidate in
                         load_legacy_scene(args.data_root, "train", scene).items()}
        elif args.union:
            retrieved = retrieve_union_scene(args.data_root, "train", scene,
                                            args.cache, workers=args.workers)
        elif args.hybrid:
            retrieved = retrieve_hybrid_scene(args.data_root, "train", scene,
                                              args.cache, workers=args.workers)
        elif args.deep:
            retrieved = retrieve_deep_scene(args.data_root, "train", scene, args.cache,
                                            workers=args.workers)
        else:
            from nocpt.retrieval import retrieve_scene
            retrieved = retrieve_scene(args.data_root, "train", scene, args.cache,
                                       workers=args.workers)
        groups = {}
        for number in range(1, int(row["n_patches"]) + 1):
            patch = cv2.imread(str(args.data_root / "train" / scene / "patches" /
                                   f"patch_{number:02}.png"), 0)
            candidates = retrieved[number]
            expected = ast.literal_eval(row[f"patch_{number:02}"])
            features = np.stack([
                np.append(feature_vector(image, patch, candidate,
                                         0 if candidate.channel == "legacy" else rank),
                          float(candidate.channel == "legacy"))
                if args.union else feature_vector(image, patch, candidate, rank)
                for rank, candidate in enumerate(candidates)])
            labels = np.asarray([
                int(expected != -1 and np.linalg.norm(
                    np.asarray((candidate.x, candidate.y)) - np.asarray(expected[:2])) <= 12)
                for candidate in candidates
            ])
            groups[number] = (candidates, features, labels)
        scene_data[scene] = groups
    cross_predictions = {}
    for held_out, groups in scene_data.items():
        train_features = np.concatenate([features for scene, data in scene_data.items()
                                         if scene != held_out for _, features, _ in data.values()])
        train_labels = np.concatenate([labels for scene, data in scene_data.items()
                                       if scene != held_out for _, _, labels in data.values()])
        model = CandidateClassifier.fit(train_features, train_labels,
                                        UNION_FEATURE_NAMES if args.union else FEATURE_NAMES)
        cross_predictions[held_out] = {
            number: (candidates, model.predict(features))
            for number, (candidates, features, _) in groups.items()
        }
    best = None
    for threshold in np.linspace(0.1, 0.9, 17):
        scores = []
        for row in truth:
            prediction = copy.deepcopy(row)
            for number, (candidates, probabilities) in cross_predictions[row["Id"]].items():
                winner = int(np.argmax(probabilities))
                candidate = candidates[winner]
                prediction[f"patch_{number:02}"] = format_cell(
                    (round(candidate.x), round(candidate.y), 0)
                    if probabilities[winner] >= threshold else None)
            scores.append(score_scene(row, prediction))
        mean = float(np.mean([item["score"] for item in scores]))
        if best is None or mean > best[0]:
            best = (mean, float(threshold), scores)
    assert best is not None
    print(f"Leave-one-scene-out mean (oracle names): {best[0]:.4f}; threshold={best[1]:.2f}")
    for row, result in zip(truth, best[2]):
        print(row["Id"], " ".join(f"{key}={value:.4f}" for key, value in result.items()))
    prevalence_scores = []
    for row in truth:
        scene = row["Id"]
        other_rows = [item for item in truth if item["Id"] != scene]
        present = sum(ast.literal_eval(item[f"patch_{number:02}"]) != -1
                      for item in other_rows
                      for number in range(1, int(item["n_patches"]) + 1))
        total = sum(int(item["n_patches"]) for item in other_rows)
        prevalence = present / total
        maxima = [float(probabilities.max())
                  for _, probabilities in cross_predictions[scene].values()]
        cutoff = max(0.40, float(np.quantile(maxima, 1-prevalence)))
        prediction = copy.deepcopy(row)
        for number, (candidates, probabilities) in cross_predictions[scene].items():
            winner = int(np.argmax(probabilities))
            candidate = candidates[winner]
            prediction[f"patch_{number:02}"] = format_cell(
                (round(candidate.x), round(candidate.y), 0)
                if probabilities[winner] >= cutoff else None)
        prevalence_scores.append(score_scene(row, prediction))
    prevalence_mean = float(np.mean([item["score"] for item in prevalence_scores]))
    print(f"Leave-one-scene-out with training class prevalence: {prevalence_mean:.4f}")
    for row, result in zip(truth, prevalence_scores):
        print(row["Id"], " ".join(f"{key}={value:.4f}" for key, value in result.items()))
    all_features = np.concatenate([features for data in scene_data.values()
                                   for _, features, _ in data.values()])
    all_labels = np.concatenate([labels for data in scene_data.values()
                                 for _, _, labels in data.values()])
    CandidateClassifier.fit(all_features, all_labels,
                            UNION_FEATURE_NAMES if args.union else FEATURE_NAMES).save(args.model)
    training_present_fraction = sum(
        ast.literal_eval(row[f"patch_{number:02}"]) != -1
        for row in truth for number in range(1, int(row["n_patches"]) + 1)
    ) / sum(int(row["n_patches"]) for row in truth)
    (args.model.parent / "threshold.json").write_text(json.dumps({
        "probability_threshold": best[1], "cross_validation_score_with_oracle_names": best[0],
        "training_present_fraction": training_present_fraction,
        "cross_validation_prevalence_score_with_oracle_names": prevalence_mean
    }, indent=2))


if __name__ == "__main__":
    main()
