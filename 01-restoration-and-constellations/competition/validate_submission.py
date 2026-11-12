"""Validate every scene, patch cell, and name in a competition CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from nocpt.data import parse_cell, read_csv


def validate(root: Path, submission: Path) -> None:
    expected_headers, expected_rows = read_csv(root / "sample_submission.csv")
    actual_headers, actual_rows = read_csv(submission)
    if actual_headers != expected_headers:
        raise ValueError("Submission header differs from sample_submission.csv")
    if len(actual_rows) != len(expected_rows):
        raise ValueError(f"Expected {len(expected_rows)} scenes, found {len(actual_rows)}")
    allowed_names = {path.name.removesuffix("_pattern.png")
                     for path in (root / "patterns").glob("*_pattern.png")}
    allowed_names.add("unknown")
    for expected, actual in zip(expected_rows, actual_rows):
        if actual["Id"] != expected["Id"] or actual["n_patches"] != expected["n_patches"]:
            raise ValueError(f"Scene identity or count mismatch: {actual['Id']}")
        if actual["constellation"] not in allowed_names:
            raise ValueError(f"Unknown constellation name: {actual['constellation']}")
        count = int(expected["n_patches"])
        for number in range(1, 88):
            key = f"patch_{number:02}"
            value = parse_cell(actual[key])
            if number > count:
                if actual[key] != expected[key]:
                    raise ValueError(f"Padding changed: {actual['Id']} {key}")
            elif value is not None:
                x, y, membership = value
                if not (0 <= x < 3000 and 0 <= y < 3000 and membership in (0, 1)):
                    raise ValueError(f"Invalid location: {actual['Id']} {key}={value}")
    print(f"Valid submission: {len(actual_rows)} scenes, {len(actual_headers)} columns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    args = parser.parse_args()
    validate(args.data_root, args.submission)


if __name__ == "__main__":
    main()
