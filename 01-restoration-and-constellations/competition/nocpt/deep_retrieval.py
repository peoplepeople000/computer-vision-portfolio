"""Optional, wider two-channel search for difficult patch correspondences."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .fitting import verify_candidate
from .matching import Candidate, coarse_search, fine_search


_IMAGE: np.ndarray | None = None
_HALF: np.ndarray | None = None


def _initialize(image_path: str) -> None:
    global _IMAGE, _HALF
    cv2.setNumThreads(1)
    _IMAGE = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if _IMAGE is None:
        raise FileNotFoundError(image_path)
    _HALF = cv2.resize(_IMAGE, (1500, 1500), interpolation=cv2.INTER_AREA)


def _worker(patch_path: str) -> list[dict]:
    assert _IMAGE is not None and _HALF is not None
    patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
    if patch is None:
        raise FileNotFoundError(patch_path)
    raw = coarse_search(_HALF, patch, keep=80)
    highpass = coarse_search(_HALF, patch, angles=36, channel="highpass", keep=40)
    refined = fine_search(_IMAGE, patch, raw + highpass, keep=60)
    for candidate in refined:
        verify_candidate(_IMAGE, patch, candidate)
    refined.sort(key=lambda item: item.confidence, reverse=True)
    return [item.serializable() for item in refined]


def _fingerprint(image_path: Path, patch_path: Path) -> str:
    digest = hashlib.sha256()
    for file in (Path(__file__), Path(__file__).with_name("matching.py"),
                 Path(__file__).with_name("fitting.py"), image_path, patch_path):
        stat = file.stat()
        digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def retrieve_deep_scene(root: Path, split: str, scene: str, cache_root: Path,
                        workers: int = 4) -> dict[int, list[Candidate]]:
    folder = root / split / scene
    image_path = folder / f"{scene}_image.png"
    patch_paths = sorted((folder / "patches").glob("patch_*.png"))
    cache_folder = cache_root / "deep" / split / scene
    cache_folder.mkdir(parents=True, exist_ok=True)
    results: dict[int, list[Candidate]] = {}
    pending: dict[int, tuple[Path, Path]] = {}
    for number, patch_path in enumerate(patch_paths, 1):
        cache_path = cache_folder / f"{patch_path.stem}_{_fingerprint(image_path, patch_path)}.json"
        if cache_path.exists():
            results[number] = [Candidate(**item) for item in json.loads(cache_path.read_text())]
        else:
            pending[number] = patch_path, cache_path
    if pending:
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                 initargs=(str(image_path),)) as executor:
            futures = {executor.submit(_worker, str(patch_path)): (number, cache_path)
                       for number, (patch_path, cache_path) in pending.items()}
            for completed, future in enumerate(as_completed(futures), 1):
                number, cache_path = futures[future]
                records = future.result()
                cache_path.write_text(json.dumps(records, indent=2))
                results[number] = [Candidate(**item) for item in records]
                if completed % 5 == 0 or completed == len(pending):
                    print(f"{scene}: deep {completed}/{len(pending)} patches, "
                          f"{time.monotonic()-started:.1f}s", flush=True)
    return dict(sorted(results.items()))
