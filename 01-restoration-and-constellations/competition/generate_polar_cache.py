"""Regenerate the preserved image-only polar/contrast candidate cache.

This optional offline stage uses the supplied scene images and query patches.
It never reads training labels, submission files, or constellation names. The
CPU implementation is slow; each completed patch is committed independently.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image


HEADER = ("x", "y", "score", "alternative_score", "angle", "scale",
          "patch_id", "new_candidate_selected")


def load_polar_module(path: Path):
    spec = importlib.util.spec_from_file_location("preserved_polar_matcher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load polar matcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def source_filename(split: str, scene: str) -> str:
    return (f"{scene}_dense_phase_predictions.csv" if split == "train"
            else f"{scene}_locations.csv")


def write_rows(path: Path, rows: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent,
                                     prefix=f".{path.stem}-", suffix=".csv",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for number in sorted(rows):
            writer.writerow(rows[number])
    os.replace(temporary, path)


def make_scene(module, data_root: Path, split: str, scene: str,
               output_root: Path, requested_patch: int | None,
               overwrite: bool) -> None:
    folder = data_root / split / scene
    image = read_gray(folder / f"{scene}_image.png")
    patch_paths = sorted((folder / "patches").glob("patch_*.png"))
    if not patch_paths:
        raise FileNotFoundError(folder / "patches")
    output = output_root / split / source_filename(split, scene)
    rows: dict[int, dict] = {}
    if output.exists():
        with output.open(newline="") as handle:
            rows = {int(row["patch_id"]): row for row in csv.DictReader(handle)}
    remaining = [(int(path.stem.split("_")[-1]), path) for path in patch_paths
                 if (requested_patch is None or int(path.stem.split("_")[-1]) == requested_patch)
                 and (overwrite or int(path.stem.split("_")[-1]) not in rows)]
    if not remaining:
        print(f"{scene}: all requested patches already cached", flush=True)
        return
    start = time.monotonic()
    search = module.build_search_index(image)
    module.add_phase_index(search)
    contrast = module.build_contrast_index(image)
    print(f"{scene}: search indexes built in {time.monotonic()-start:.1f}s", flush=True)
    for position, (number, path) in enumerate(remaining, 1):
        patch = read_gray(path)
        retrieved = (
            module.coarse_candidates(search, patch, top_k=4096)[0],
            module.phase_candidates(search, patch, top_k=4096)[0],
            module.contrast_candidates(contrast, patch, top_k=4096)[0],
        )
        # The saved dense-phase stage pooled three image-only retrievals.
        centers = module.torch.unique(module.torch.cat(retrieved), dim=0)
        result = module.dense_contrast_match(contrast, patch, centers)
        rows[number] = {**result, "patch_id": number,
                        "new_candidate_selected": False}
        write_rows(output, rows)
        print(f"{scene}: {position}/{len(remaining)} patches, "
              f"elapsed {time.monotonic()-start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--scene", help="Process one scene; omit to process the full split")
    parser.add_argument("--patch", type=int, help="Process one patch in a selected scene")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reference/legacy_dense"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing image-derived cache rows")
    args = parser.parse_args()
    if args.patch is not None and args.scene is None:
        parser.error("--patch requires --scene")
    module = load_polar_module(Path("reference/legacy_dense/q5_experimental_functions.py"))
    scenes = ([args.scene] if args.scene else
              sorted(path.name for path in (args.data_root / args.split).iterdir()
                     if path.is_dir()))
    for scene in scenes:
        make_scene(module, args.data_root, args.split, scene, args.output_root,
                   args.patch, args.overwrite)


if __name__ == "__main__":
    main()
