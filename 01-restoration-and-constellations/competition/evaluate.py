"""Score predictions against the three labelled training scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

from nocpt.data import read_csv
from nocpt.scoring import score_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()
    _, truth = read_csv(args.data_root / "train_ground_truth.csv")
    _, predictions = read_csv(args.predictions)
    index = {row["Id"]: row for row in predictions}
    results = []
    for row in truth:
        if args.scene and row["Id"] != args.scene:
            continue
        if row["Id"] not in index:
            raise ValueError(f"Missing scene {row['Id']}")
        scores = score_scene(row, index[row["Id"]])
        results.append(scores)
        print(row["Id"], " ".join(f"{key}={value:.4f}" for key, value in scores.items()))
    print("mean", " ".join(f"{key}={sum(row[key] for row in results)/len(results):.4f}"
                             for key in results[0]))


if __name__ == "__main__":
    main()
