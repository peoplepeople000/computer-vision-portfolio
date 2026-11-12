"""Dataset and submission parsing."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def parse_cell(value: str) -> tuple[int, int, int] | None:
    value = value.strip()
    if value == "-1":
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = tuple(int(token) for token in value.replace(",", " ").split())
    if not isinstance(parsed, (tuple, list)) or len(parsed) not in (2, 3):
        raise ValueError(f"Invalid patch value: {value!r}")
    x, y = map(int, parsed[:2])
    membership = int(parsed[2]) if len(parsed) == 3 else 0
    if membership not in (0, 1):
        raise ValueError(f"Invalid membership: {value!r}")
    return x, y, membership


def load_scene(root: Path, split: str, name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    folder = root / split / name
    image = cv2.imread(str(folder / f"{name}_image.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(folder / f"{name}_image.png")
    patch_paths = sorted((folder / "patches").glob("patch_*.png"))
    patches = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in patch_paths]
    if any(patch is None for patch in patches):
        raise ValueError(f"Unreadable patch in {folder}")
    return image, patches


def scene_rows(root: Path, split: str) -> tuple[list[str], list[dict[str, str]]]:
    filename = "train_ground_truth.csv" if split == "train" else "sample_submission.csv"
    headers, rows = read_csv(root / filename)
    for row in rows:
        folder = root / split / row["Id"] / "patches"
        actual = len(list(folder.glob("patch_*.png")))
        if actual != int(row["n_patches"]):
            raise ValueError(f"Patch count mismatch for {row['Id']}: {actual}")
    return headers, rows


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: tuple[int, int, int] | None) -> str:
    return "-1" if value is None else f"({value[0]}, {value[1]}, {value[2]})"
