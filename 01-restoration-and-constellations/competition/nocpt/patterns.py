"""Extract white star nodes from provided RGBA diagrams."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def extract_nodes(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA pattern: {path}")
    b, g, r, alpha = cv2.split(image)
    white = ((b >= 175) & (g >= 175) & (r >= 175) & (alpha >= 100)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(white, 8)
    nodes = []
    for index in range(1, count):
        if 3 <= stats[index, cv2.CC_STAT_AREA] <= 180:
            nodes.append(centroids[index])
    if len(nodes) < 2:
        raise ValueError(f"Too few nodes in {path}: {len(nodes)}")
    return np.asarray(nodes, dtype=np.float64)


def load_patterns(root: Path) -> dict[str, np.ndarray]:
    paths = sorted((root / "patterns").glob("*_pattern.png"))
    if len(paths) != 48:
        raise ValueError(f"Expected 48 patterns, found {len(paths)}")
    return {path.name.removesuffix("_pattern.png"): extract_nodes(path) for path in paths}
