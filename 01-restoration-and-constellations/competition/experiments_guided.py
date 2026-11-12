"""Audit a second, deterministic image-only constellation geometry method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

from nocpt.classifier import CandidateClassifier, UNION_FEATURE_NAMES
from nocpt.patterns import load_patterns
from nocpt.union_retrieval import retrieve_union_scene


sys.path.insert(0, str(Path(__file__).parent / "reference" / "legacy_geometry"))
from joint_ransac import JointConfig  # noqa: E402
from joint_ransac_guided import GuidedConfig, fit_joint_constellations_guided  # noqa: E402


SCENES = ("pisces", "scorpius", "taurus")


def scene_model(scene: str, split: str) -> CandidateClassifier:
    if split == "validation":
        return CandidateClassifier.load(Path("models/union/candidate.json"))
    arrays = np.load("outputs/experiments/union_features.npz")
    train = arrays["sceneids"] != SCENES.index(scene)
    features = np.column_stack((arrays["features"], arrays["legacy"]))
    return CandidateClassifier.fit(features[train], arrays["labels"][train],
                                   UNION_FEATURE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--patch-pairs", type=int, default=64)
    parser.add_argument("--exact", type=int, default=40)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/experiments/guided"))
    parser.add_argument("--force", action="store_true",
                        help="Recompute rankings already saved for a scene")
    args = parser.parse_args()
    root = Path("participant")
    patterns = load_patterns(root)
    scenes = ([args.scene] if args.scene else
              list(SCENES) if args.split == "train" else
              sorted(p.name for p in (root / "validation").iterdir() if p.is_dir()))
    config = GuidedConfig(joint=JointConfig(max_candidates_per_patch=5),
                          patch_pairs=args.patch_pairs, exact_hypotheses=args.exact)
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        path = out / f"{args.split}_{scene}_rankings.csv"
        if path.exists() and not args.force:
            print(args.split, scene, "cached", flush=True)
            continue
        started = time.monotonic()
        model = scene_model(scene, args.split)
        image = cv2.imread(str(root / args.split / scene / f"{scene}_image.png"), 0)
        retrieved = retrieve_union_scene(root, args.split, scene,
                                         Path("outputs/cache"))
        records = []
        for number, choices in retrieved.items():
            patch = cv2.imread(str(root / args.split / scene / "patches" /
                                   f"patch_{number:02d}.png"), 0)
            features = np.stack([model.extract(image, patch, c, rank)
                                 for rank, c in enumerate(choices)])
            probabilities = model.predict(features)
            for candidate, probability in zip(choices, probabilities):
                records.append({"patch_id": number, "x": candidate.x,
                                "y": candidate.y, "appearance_score": float(probability),
                                "presence_prob": float(probability),
                                "source": "saved_dense_phase" if candidate.channel == "legacy"
                                else "retrieved"})
        result = fit_joint_constellations_guided(pd.DataFrame(records), patterns,
                                                 config=config)
        result.rankings.to_csv(path, index=False)
        if result.best_pose is not None:
            pose = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in result.best_pose.items()}
            (out / f"{args.split}_{scene}_pose.json").write_text(
                json.dumps(pose, indent=2))
        result.selections.to_csv(out / f"{args.split}_{scene}_selections.csv",
                                 index=False)
        print(args.split, scene, result.rankings.iloc[0]["pattern"],
              result.rankings.iloc[0]["score"],
              "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__":
    main()
