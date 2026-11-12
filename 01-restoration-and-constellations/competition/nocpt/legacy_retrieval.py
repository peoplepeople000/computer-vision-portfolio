"""Load independently generated image-only candidates from the earlier pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2

from .fitting import verify_candidate
from .matching import Candidate, fine_search


def _fingerprint(source: Path, image: Path, patches: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in (source, image, Path(__file__),
                 Path(__file__).with_name("matching.py"),
                 Path(__file__).with_name("fitting.py"), *patches):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def load_legacy_scene(data_root: Path, split: str, scene: str,
                      root: Path = Path("reference/legacy_dense")) -> dict[int, Candidate]:
    """Refine the saved legacy image matches without reading ground truth."""
    directory = root / split
    filename = (f"{scene}_dense_phase_predictions.csv" if split == "train"
                else f"{scene}_locations.csv")
    source = directory / filename
    cache = directory / f"{scene}_current_fit.json"
    folder = data_root / split / scene
    image_path = folder / f"{scene}_image.png"
    patch_paths = sorted((folder / "patches").glob("patch_*.png"))
    fingerprint = _fingerprint(source, image_path, patch_paths)
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("fingerprint") == fingerprint:
            result = {int(k): Candidate(**v)
                      for k, v in cached["candidates"].items()}
            for candidate in result.values():
                candidate.channel = "legacy"
            return result
    image = cv2.imread(str(image_path), 0)
    if image is None:
        raise FileNotFoundError(scene)
    result: dict[int, Candidate] = {}
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        number = int(row["patch_id"])
        patch = cv2.imread(str(data_root / split / scene / "patches" /
                               f"patch_{number:02d}.png"), 0)
        candidate = Candidate(float(row["x"]), float(row["y"]),
                              -float(row["angle"]) % 360,
                              float(row["scale"]), float(row["score"]), "raw")
        refined = fine_search(image, patch, [candidate], keep=1)
        if refined:
            candidate = verify_candidate(image, patch, refined[0])
        candidate.channel = "legacy"
        result[number] = candidate
    payload = json.dumps({
        "fingerprint": fingerprint,
        "candidates": {str(k): v.serializable() for k, v in result.items()},
    }, indent=2)
    with tempfile.NamedTemporaryFile("w", dir=cache.parent,
                                     prefix=f".{cache.stem}-", suffix=".json",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, cache)
    return result
