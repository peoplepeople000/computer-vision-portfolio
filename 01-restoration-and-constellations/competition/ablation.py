"""Compare localization stages on the three labelled scenes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from nocpt.classifier import CandidateClassifier, feature_vector
from nocpt.data import format_cell, read_csv
from nocpt.matching import Candidate
from nocpt.scoring import score_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--final", type=Path, default=Path("outputs/train_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments.json"))
    parser.add_argument("--model", type=Path, default=Path("models/candidate.json"))
    args = parser.parse_args()
    model = CandidateClassifier.load(args.model)
    threshold = json.loads((args.model.parent / "threshold.json").read_text())[
        "probability_threshold"]
    _, truths = read_csv(args.data_root / "train_ground_truth.csv")
    _, finals = read_csv(args.final)
    final_by_scene = {row["Id"]: row for row in finals}
    all_scores: dict[str, list[dict]] = {name: [] for name in (
        "ncc_only", "verifier_raw", "verifier_with_highpass", "full_pipeline")}
    for truth in truths:
        scene = truth["Id"]
        image = cv2.imread(str(args.data_root / "train" / scene / f"{scene}_image.png"), 0)
        cache = json.loads((args.diagnostics / f"{scene}_candidates.json").read_text())[
            "patches"]
        guesses = {name: copy.deepcopy(truth) for name in all_scores}
        for number in range(1, int(truth["n_patches"]) + 1):
            key = f"patch_{number:02}"
            patch = cv2.imread(str(args.data_root / "train" / scene / "patches" /
                                   f"{key}.png"), 0)
            candidates = [Candidate(**item) for item in cache[key]]
            raw = [candidate for candidate in candidates if candidate.channel == "raw"]
            if raw:
                winner = max(raw, key=lambda candidate: candidate.confidence)
                guesses["ncc_only"][key] = format_cell(
                    (round(winner.x), round(winner.y), 0)
                    if winner.confidence >= 0.9 else None)
            else:
                guesses["ncc_only"][key] = "-1"
            for name, pool in (("verifier_raw", raw),
                               ("verifier_with_highpass", candidates)):
                if not pool:
                    guesses[name][key] = "-1"
                    continue
                features = np.stack([feature_vector(image, patch, item, rank)
                                     for rank, item in enumerate(pool)])
                probabilities = model.predict(features)
                index = int(np.argmax(probabilities))
                winner = pool[index]
                guesses[name][key] = format_cell(
                    (round(winner.x), round(winner.y), 0)
                    if probabilities[index] >= threshold else None)
        guesses["full_pipeline"] = final_by_scene[scene]
        for name, guess in guesses.items():
            # Fix the name across retrieval ablations to compare only patch decisions.
            if name != "full_pipeline":
                guess["constellation"] = truth["constellation"]
            all_scores[name].append({"scene": scene, **score_scene(truth, guess)})
    report = {name: {
        "mean": {key: float(np.mean([row[key] for row in rows]))
                 for key in ("presence", "localization", "geometric_recovery",
                             "identification", "score")},
        "scenes": rows,
    } for name, rows in all_scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for name, value in report.items():
        print(name, " ".join(f"{key}={score:.4f}" for key, score in value["mean"].items()))


if __name__ == "__main__":
    main()
