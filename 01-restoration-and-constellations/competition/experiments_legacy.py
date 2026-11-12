"""Cross-validate a union of current and legacy image-only candidate banks."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nocpt.classifier import feature_vector
from nocpt.data import format_cell, parse_cell
from nocpt.legacy_retrieval import load_legacy_scene
from nocpt.scoring import score_scene


SCENES = ("pisces", "scorpius", "taurus")
ROOT = Path("participant")


def augmented_data() -> dict[str, np.ndarray]:
    current = np.load("outputs/experiments/deep_features.npz")
    arrays = {key: current[key] for key in current.files}
    old_features = []
    old_labels = []
    old_sceneids = []
    old_patchids = []
    old_present = []
    old_xs = []
    old_ys = []
    ground_truth = {r["Id"]: r for r in csv.DictReader(open(ROOT / "train_ground_truth.csv"))}
    for scene_id, scene in enumerate(SCENES):
        image = cv2.imread(str(ROOT / "train" / scene / f"{scene}_image.png"), 0)
        candidates = load_legacy_scene(ROOT, "train", scene)
        for number, candidate in candidates.items():
            patch = cv2.imread(str(ROOT / "train" / scene / "patches" /
                                   f"patch_{number:02d}.png"), 0)
            target = parse_cell(ground_truth[scene][f"patch_{number:02d}"])
            old_features.append(feature_vector(image, patch, candidate, 0))
            old_labels.append(int(target is not None and
                                  np.hypot(candidate.x-target[0], candidate.y-target[1]) <= 12))
            old_present.append(int(target is not None))
            old_sceneids.append(scene_id)
            old_patchids.append(number)
            old_xs.append(candidate.x)
            old_ys.append(candidate.y)
        print(f"Legacy features: {scene}", flush=True)
    arrays["features"] = np.vstack((arrays["features"], old_features))
    for name, values in (
        ("labels", old_labels), ("sceneids", old_sceneids),
        ("patchids", old_patchids), ("present", old_present),
        ("xs", old_xs), ("ys", old_ys),
        ("ranks", [-1]*len(old_labels)),
    ):
        arrays[name] = np.concatenate((arrays[name], np.asarray(values)))
    arrays["legacy"] = np.concatenate((np.zeros(len(current["labels"]), dtype=int),
                                       np.ones(len(old_labels), dtype=int)))
    np.savez("outputs/experiments/union_features.npz", **arrays)
    return arrays


def benchmark(arrays: dict[str, np.ndarray]) -> None:
    truth = {r["Id"]: r for r in csv.DictReader(open(ROOT / "train_ground_truth.csv"))}
    model_constructors = {
        "logistic": lambda: LogisticRegression(C=0.35, class_weight="balanced",
                                                 max_iter=1000, random_state=6643),
        "forest": lambda: RandomForestClassifier(n_estimators=400,
                                                    min_samples_leaf=4,
                                                    max_features=0.6,
                                                    class_weight="balanced",
                                                    random_state=6643, n_jobs=4),
        "histgb": lambda: HistGradientBoostingClassifier(max_iter=200,
                                                           max_leaf_nodes=7,
                                                           min_samples_leaf=12,
                                                           l2_regularization=3,
                                                           random_state=6643),
    }
    for variant in ("deep", "old", "union"):
        mask = ((arrays["legacy"] == 0) if variant == "deep" else
                (arrays["legacy"] == 1) if variant == "old" else
                np.ones(len(arrays["labels"]), dtype=bool))
        for model_name, constructor in model_constructors.items():
            scores = np.zeros(len(arrays["labels"]))
            features = np.column_stack((arrays["features"], arrays["legacy"]))
            for held in range(3):
                training = mask & (arrays["sceneids"] != held)
                testing = mask & (arrays["sceneids"] == held)
                scaler = StandardScaler() if model_name == "logistic" else None
                x_train = (scaler.fit_transform(features[training]) if scaler else
                           features[training])
                x_test = scaler.transform(features[testing]) if scaler else features[testing]
                model = constructor()
                model.fit(x_train, arrays["labels"][training])
                scores[testing] = model.predict_proba(x_test)[:, 1]
            top1 = np.zeros(3, int)
            means = []
            for threshold in np.arange(0.10, 0.91, 0.05):
                scene_scores = []
                for scene_id, scene in enumerate(SCENES):
                    actual = truth[scene]
                    predicted = actual.copy()
                    for number in range(1, int(actual["n_patches"]) + 1):
                        rows = np.where(mask & (arrays["sceneids"] == scene_id) &
                                        (arrays["patchids"] == number))[0]
                        if not len(rows):
                            continue
                        winner = rows[int(np.argmax(scores[rows]))]
                        if threshold == 0.10 and arrays["present"][winner]:
                            top1[scene_id] += arrays["labels"][winner]
                        predicted[f"patch_{number:02d}"] = format_cell(
                            (round(arrays["xs"][winner]), round(arrays["ys"][winner]), 0)
                            if scores[winner] >= threshold else None)
                    scene_scores.append(score_scene(actual, predicted)["score"])
                means.append((float(np.mean(scene_scores)), threshold, scene_scores))
            best = max(means)
            print(variant, model_name, "top1", top1.tolist(),
                  "best_oracle_name", round(best[0], 4), "threshold", round(best[1], 2),
                  "per_scene", [round(x, 4) for x in best[2]], flush=True)


if __name__ == "__main__":
    path = Path("outputs/experiments/union_features.npz")
    arrays = {k: v for k, v in np.load(path).items()} if path.exists() else augmented_data()
    benchmark(arrays)
