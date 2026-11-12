"""Stress-test held-out candidate verification under extra query degradation.

Retrieval positions are frozen, so this tests verification and local
re-refinement, not global search recall. Training labels are used only to
measure the final predictions, never to select a candidate or transform.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from nocpt.classifier import CandidateClassifier, UNION_FEATURE_NAMES, feature_vector
from nocpt.data import format_cell, read_csv
from nocpt.fitting import verify_candidate
from nocpt.matching import Candidate, fine_search
from nocpt.scoring import score_scene
from nocpt.union_retrieval import retrieve_union_scene


SCENES = ("pisces", "scorpius", "taurus")


def degrade(patch: np.ndarray, severity: float, seed: int) -> np.ndarray:
    if severity == 0:
        return patch.copy()
    rng = np.random.default_rng(seed)
    image = patch.astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 0.8*severity)
    gradient = np.linspace(-1, 1, image.shape[1], dtype=np.float32)[None]
    image = image * (1 + 0.12*severity*gradient)
    image += rng.normal(0, 13*severity, image.shape).astype(np.float32)
    impulse = rng.random(image.shape)
    image[impulse < 0.0025*severity] = 0
    image[impulse > 1-0.0025*severity] = 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    quality = max(35, int(75-20*severity))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG perturbation failed")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def held_out_model(arrays: np.lib.npyio.NpzFile, scene_id: int) -> CandidateClassifier:
    features = np.column_stack((arrays["features"], arrays["legacy"]))
    selected = arrays["sceneids"] != scene_id
    return CandidateClassifier.fit(features[selected], arrays["labels"][selected],
                                   UNION_FEATURE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/union_stress.csv"))
    parser.add_argument("--severity", type=float, nargs="+", default=[0, 0.7, 1.0])
    parser.add_argument("--min-present-fraction", type=float, default=0.0)
    args = parser.parse_args()
    _, truth_rows = read_csv(Path("participant/train_ground_truth.csv"))
    truth = {row["Id"]: row for row in truth_rows}
    arrays = np.load("outputs/experiments/union_features.npz")
    results = []
    for scene_id, scene in enumerate(SCENES):
        model = held_out_model(arrays, scene_id)
        folder = Path("participant/train") / scene
        image = cv2.imread(str(folder / f"{scene}_image.png"), 0)
        bank = retrieve_union_scene(Path("participant"), "train", scene,
                                    Path("outputs/cache"))
        for severity in args.severity:
            actual = truth[scene]
            predicted = actual.copy()
            count = 0
            proposals: dict[int, tuple[float, Candidate]] = {}
            for number in range(1, int(actual["n_patches"]) + 1):
                patch = cv2.imread(str(folder / "patches" / f"patch_{number:02}.png"), 0)
                query = degrade(patch, severity, 6643+1000*scene_id+number)
                original = bank[number]
                # Fixed clean-search shortlist; no ground-truth localization used.
                clean = np.stack([model.extract(image, patch, item, rank)
                                  for rank, item in enumerate(original)])
                order = np.argsort(-model.predict(clean), kind="stable")[:10]
                old_index = len(original)-1
                selected_indices = sorted(set([*order, old_index]))
                alternatives = []
                for index in selected_indices:
                    saved = original[index]
                    candidate = Candidate(**saved.serializable())
                    candidate.channel = "raw" if saved.channel == "legacy" else saved.channel
                    refined = fine_search(image, query, [candidate], keep=1)
                    if not refined:
                        continue
                    candidate = verify_candidate(image, query, refined[0])
                    candidate.channel = saved.channel
                    vector = model.extract(image, query, candidate, index)
                    probability = float(model.predict(vector[None])[0])
                    alternatives.append((probability, candidate))
                if alternatives:
                    probability, winner = max(alternatives, key=lambda item: item[0])
                    proposals[number] = (probability, winner)
                    if probability >= 0.75:
                        predicted[f"patch_{number:02d}"] = format_cell(
                            (round(winner.x), round(winner.y), 0))
                        count += 1
                    else:
                        predicted[f"patch_{number:02d}"] = format_cell(None)
                else:
                    predicted[f"patch_{number:02d}"] = format_cell(None)
            if args.min_present_fraction:
                minimum = int(np.ceil(args.min_present_fraction * int(actual["n_patches"])))
                selected = [proposals[number][1] for number in proposals
                            if predicted[f"patch_{number:02d}"] != "-1"]
                remaining = sorted(((probability, number, candidate)
                                    for number, (probability, candidate) in proposals.items()
                                    if predicted[f"patch_{number:02d}"] == "-1"),
                                   reverse=True, key=lambda entry: entry[0])
                for probability, number, candidate in remaining:
                    if count >= minimum or probability < 0.05:
                        break
                    if any(np.hypot(candidate.x-other.x, candidate.y-other.y) < 12
                           for other in selected):
                        continue
                    selected.append(candidate)
                    predicted[f"patch_{number:02d}"] = format_cell(
                        (round(candidate.x), round(candidate.y), 0))
                    count += 1
            score = score_scene(actual, predicted)
            results.append({"scene": scene, "severity": severity,
                            "present_count": count, **score})
            print(scene, severity, count, score["score"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
