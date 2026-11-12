"""Evaluate query-frame photometric evidence on cached training candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nocpt.matching import Candidate
from nocpt.patch_evidence import evidence_vector


ROOT = Path("participant")
SCENES = ("pisces", "scorpius", "taurus")


def evidence_matrix(source: Path) -> np.ndarray:
    dataset = np.load(source)
    result = np.zeros((len(dataset["labels"]), 11), dtype=np.float64)
    for scene_id, scene in enumerate(SCENES):
        folder = ROOT / "train" / scene
        image = cv2.imread(str(folder / f"{scene}_image.png"), 0)
        for patch_id in np.unique(dataset["patchids"][dataset["sceneids"] == scene_id]):
            patch = cv2.imread(str(folder / "patches" / f"patch_{patch_id:02}.png"), 0)
            cache_path = next((Path("outputs/cache/deep/train") / scene).glob(f"patch_{patch_id:02}_*.json"))
            candidates = [Candidate(**item) for item in json.loads(cache_path.read_text())]
            rows = np.where((dataset["sceneids"] == scene_id) & (dataset["patchids"] == patch_id))[0]
            for row in rows:
                rank = int(dataset["ranks"][row])
                candidate = candidates[rank]
                if abs(candidate.x - dataset["xs"][row]) > 0.1 or abs(candidate.y - dataset["ys"][row]) > 0.1:
                    raise ValueError("Candidate order mismatch")
                result[row] = evidence_vector(image, patch, candidate)
        print(f"Finished evidence for {scene}", flush=True)
    return result


def benchmark(source: Path, evidence: np.ndarray) -> None:
    data = np.load(source)
    old = data["features"]
    labels = data["labels"]
    sceneids = data["sceneids"]
    patchids = data["patchids"]
    present = data["present"]
    groups = [(scene_id, patch_id) for scene_id in range(3)
              for patch_id in np.unique(patchids[sceneids == scene_id])]
    variants = {
        "old": old,
        "evidence": evidence,
        "combined": np.column_stack((old, evidence)),
        "combined_norank": np.column_stack((old[:, :12], old[:, 13:], evidence)),
    }
    models = {
        "logistic": lambda: LogisticRegression(C=0.35, class_weight="balanced", max_iter=1000),
        "forest": lambda: RandomForestClassifier(n_estimators=300, max_features=0.6,
                                                    min_samples_leaf=5, class_weight="balanced",
                                                    random_state=6643, n_jobs=4),
        "histgb": lambda: HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=7,
                                                           min_samples_leaf=12, l2_regularization=3,
                                                           random_state=6643),
    }
    for variant, features in variants.items():
        for model_name, constructor in models.items():
            successes = np.zeros(3, dtype=int)
            for held_out in range(3):
                train = sceneids != held_out
                test = ~train
                scaler = StandardScaler() if model_name == "logistic" else None
                x_train = scaler.fit_transform(features[train]) if scaler else features[train]
                x_test = scaler.transform(features[test]) if scaler else features[test]
                model = constructor()
                model.fit(x_train, labels[train])
                probs = np.zeros(len(labels))
                probs[test] = model.predict_proba(x_test)[:, 1]
                for scene_id, patch_id in groups:
                    if scene_id != held_out:
                        continue
                    rows = np.where((sceneids == scene_id) & (patchids == patch_id))[0]
                    if present[rows[0]]:
                        successes[scene_id] += labels[rows[np.argmax(probs[rows])]]
            print(variant, model_name, successes.tolist(), int(sum(successes)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/experiments/deep_features.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/deep_evidence.npy"))
    args = parser.parse_args()
    if args.output.exists():
        evidence = np.load(args.output)
    else:
        evidence = evidence_matrix(args.input)
        np.save(args.output, evidence)
    benchmark(args.input, evidence)


if __name__ == "__main__":
    main()
