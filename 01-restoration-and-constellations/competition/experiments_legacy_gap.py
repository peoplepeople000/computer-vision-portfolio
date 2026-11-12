"""Ablate image-only dense-search score and runner-up gap features."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from nocpt.classifier import CandidateClassifier, UNION_FEATURE_NAMES
from nocpt.data import format_cell, read_csv
from nocpt.scoring import score_scene


SCENES = ("pisces", "scorpius", "taurus")


def dense_features(split: str, scene: str) -> dict[int, tuple[float, float]]:
    name = (f"{scene}_dense_phase_predictions.csv" if split == "train"
            else f"{scene}_locations.csv")
    path = Path("reference/legacy_dense") / split / name
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["patch_id"]):
            (float(row["score"]),
             float(row["score"]) - float(row["alternative_score"]))
            for row in rows}


def main() -> None:
    arrays = np.load("outputs/experiments/union_features.npz")
    basic = np.column_stack((arrays["features"], arrays["legacy"]))
    dense = [dense_features("train", scene) for scene in SCENES]
    extra = np.asarray([dense[int(scene)][int(patch)]
                        for scene, patch in zip(arrays["sceneids"], arrays["patchids"])])
    _, truth_rows = read_csv(Path("participant/train_ground_truth.csv"))
    truth = {row["Id"]: row for row in truth_rows}
    for variant, features, names in (
        ("basic", basic, UNION_FEATURE_NAMES),
        ("dense_score", np.column_stack((basic, extra[:, :1])),
         UNION_FEATURE_NAMES + ("dense_score",)),
        ("dense_gap", np.column_stack((basic, extra[:, 1:])),
         UNION_FEATURE_NAMES + ("dense_gap",)),
        ("both", np.column_stack((basic, extra)),
         UNION_FEATURE_NAMES + ("dense_score", "dense_gap")),
    ):
        probabilities = np.zeros(len(features))
        for held in range(3):
            train = arrays["sceneids"] != held
            model = CandidateClassifier.fit(features[train], arrays["labels"][train], names)
            probabilities[~train] = model.predict(features[~train])
        patch_probs, patch_labels = [], []
        print(variant, flush=True)
        for threshold in (0.55, 0.65, 0.75, 0.85):
            results = []
            accepted = []
            top1 = []
            for scene_id, scene in enumerate(SCENES):
                actual = truth[scene]
                predicted = actual.copy()
                accepted_here = 0
                top1_here = 0
                for number in range(1, int(actual["n_patches"]) + 1):
                    rows = np.where((arrays["sceneids"] == scene_id) &
                                    (arrays["patchids"] == number))[0]
                    winner = rows[np.argmax(probabilities[rows])]
                    if threshold == 0.55:
                        patch_probs.append(float(probabilities[winner]))
                        patch_labels.append(int(arrays["present"][winner]))
                    accepted_here += int(probabilities[winner] >= threshold)
                    top1_here += int(arrays["labels"][winner])
                    predicted[f"patch_{number:02d}"] = format_cell(
                        (round(arrays["xs"][winner]), round(arrays["ys"][winner]), 0)
                        if probabilities[winner] >= threshold else None)
                accepted.append(accepted_here)
                top1.append(top1_here)
                results.append(score_scene(actual, predicted)["score"])
            print("threshold", threshold, "score", round(float(np.mean(results)), 4),
                  "per_scene", [round(s, 4) for s in results],
                  "accepted", accepted, "top1", top1, flush=True)
        print("patch_auc", round(roc_auc_score(patch_labels, patch_probs), 4),
              flush=True)


if __name__ == "__main__":
    main()
