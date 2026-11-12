"""Combine two independently generated image-only candidate searches."""

from __future__ import annotations

from pathlib import Path

from .deep_retrieval import retrieve_deep_scene
from .legacy_retrieval import load_legacy_scene
from .matching import Candidate


def retrieve_union_scene(root: Path, split: str, scene: str,
                         cache_root: Path, workers: int = 4) -> dict[int, list[Candidate]]:
    current = retrieve_deep_scene(root, split, scene, cache_root, workers)
    legacy = load_legacy_scene(root, split, scene)
    if current.keys() != legacy.keys():
        raise ValueError(f"Candidate bank mismatch for {scene}")
    return {number: choices + [legacy[number]] for number, choices in current.items()}
