"""Apply one fixed, evidence-gated second geometry opinion to a submission.

The guide is computed from image-only candidates by experiments_guided.py.
This script never consults validation labels or any scored submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from nocpt.data import format_cell, parse_cell, read_csv, write_csv
from nocpt.patterns import load_patterns


RULE = {
    "minimum_adjusted_guide_score": 2.0,
    "minimum_guide_margin": 1.0,
    "minimum_guide_support": 5,
    "minimum_guide_coverage": 0.5,
    "primary_weak_if_matches_below": 4,
    "primary_weak_if_margin_below": 3.0,
}


def read_rankings(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def primary_evidence(path: Path, name: str) -> tuple[float, int]:
    rankings = json.loads(path.read_text())["constellation_ranking"]
    chosen = next((row for row in rankings if row["name"] == name), None)
    if chosen is None:
        return float("-inf"), 0
    competitor = max((float(row["score"]) for row in rankings
                      if row["name"] != name), default=float("-inf"))
    return float(chosen["score"])-competitor, int(chosen["matched"])


def should_use_guide(primary_name: str, primary_margin: float,
                     primary_matches: int, rankings: list[dict[str, str]]) -> bool:
    """Require a multi-node, null-adjusted lead when the primary fit is weak.

    These global thresholds use geometric support rather than scene identity
    or leaderboard feedback. The guide remains advisory for ambiguous fits.
    """
    if len(rankings) < 2 or rankings[0]["pattern"] == primary_name:
        return False
    leader = rankings[0]
    guide_margin = float(leader["score"])-float(rankings[1]["score"])
    strong_guide = (
        float(leader["score"]) >= RULE["minimum_adjusted_guide_score"]
        and guide_margin >= RULE["minimum_guide_margin"]
        and int(leader["support"]) >= RULE["minimum_guide_support"]
        and float(leader["coverage_fraction"]) >= RULE["minimum_guide_coverage"]
    )
    weak_primary = (
        primary_matches < RULE["primary_weak_if_matches_below"]
        or primary_margin < RULE["primary_weak_if_margin_below"]
    )
    return strong_guide and weak_primary


def update_membership(row: dict[str, str], nodes: np.ndarray,
                      pose: dict) -> None:
    matrix = np.asarray(pose["matrix"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64)
    projected = nodes @ matrix.T + translation
    tree = cKDTree(projected)
    for number in range(1, int(row["n_patches"]) + 1):
        key = f"patch_{number:02d}"
        cell = parse_cell(row[key])
        if cell is not None:
            x, y, _ = cell
            row[key] = format_cell((x, y, int(tree.query([x, y])[0] <= 24)))


def render_overlay(row: dict[str, str], projected: np.ndarray,
                   image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in projected:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(canvas, (round(x), round(y)), 13, (0, 220, 0), 2)
    for number in range(1, int(row["n_patches"]) + 1):
        cell = parse_cell(row[f"patch_{number:02d}"])
        if cell is not None:
            cv2.circle(canvas, cell[:2], 7, (0, 0, 255), 2)
    canvas = cv2.resize(canvas, (1500, 1500), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"Guided pattern: {row['constellation']}", (24, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("outputs/submission_union_robust.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/submission_union_guided.csv"))
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("outputs/submission_union_robust_diagnostics/results"))
    parser.add_argument("--guided", type=Path,
                        default=Path("outputs/experiments/guided"))
    parser.add_argument("--split", choices=("train", "validation"),
                        default="validation")
    args = parser.parse_args()
    headers, rows = read_csv(args.base)
    patterns = load_patterns(Path("participant"))
    audit = []
    for row in rows:
        scene = row["Id"]
        primary_name = row["constellation"]
        primary_margin, primary_matches = primary_evidence(
            args.diagnostics / f"{scene}.json", primary_name)
        guide = read_rankings(args.guided / f"{args.split}_{scene}_rankings.csv")
        guide_margin = (float(guide[0]["score"])-float(guide[1]["score"])
                        if len(guide) >= 2 else float("nan"))
        switched = should_use_guide(primary_name, primary_margin,
                                    primary_matches, guide)
        if switched:
            pose_path = args.guided / f"{args.split}_{scene}_pose.json"
            pose = json.loads(pose_path.read_text())
            if pose["pattern"] != guide[0]["pattern"]:
                raise ValueError(f"Guided pose and ranking disagree for {scene}")
            row["constellation"] = pose["pattern"]
            update_membership(row, patterns[pose["pattern"]], pose)
            matrix = np.asarray(pose["matrix"], dtype=np.float64)
            translation = np.asarray(pose["translation"], dtype=np.float64)
            projected = patterns[pose["pattern"]] @ matrix.T + translation
            render_overlay(
                row, projected,
                Path("participant") / args.split / scene / f"{scene}_image.png",
                args.output.parent / f"{args.output.stem}_diagnostics" /
                f"{scene}_guided_overlay.png",
            )
        audit.append({
            "scene": scene, "primary_name": primary_name,
            "primary_margin": primary_margin,
            "primary_matches": primary_matches,
            "guided_name": guide[0]["pattern"],
            "guided_score": guide[0]["score"],
            "guided_margin": guide_margin,
            "guided_support": guide[0]["support"],
            "guided_coverage": guide[0]["coverage_fraction"],
            "switched": switched,
        })
    partial_output = args.output.with_suffix(".partial.csv")
    write_csv(partial_output, headers, rows)
    os.replace(partial_output, args.output)
    audit_path = args.output.with_suffix(".audit.csv")
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit[0].keys())
        writer.writeheader()
        writer.writerows(audit)
    fingerprint = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "base_csv_path": str(args.base),
        "base_csv_sha256": fingerprint(args.base),
        "output_csv_sha256": fingerprint(args.output),
        "guided_ranking_sha256": {
            row["Id"]: fingerprint(args.guided /
                                   f"{args.split}_{row['Id']}_rankings.csv")
            for row in rows
        },
        "switched_scenes": [item["scene"] for item in audit if item["switched"]],
        "rule": RULE,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.output}; changed {sum(item['switched'] for item in audit)} names")


if __name__ == "__main__":
    main()
