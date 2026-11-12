"""Augment the stable shortlist with a few high-quality deep-search locations."""

from __future__ import annotations

from pathlib import Path
import math

from .deep_retrieval import retrieve_deep_scene
from .matching import Candidate
from .retrieval import retrieve_scene


def retrieve_hybrid_scene(root: Path, split: str, scene: str, cache_root: Path,
                          workers: int = 4) -> dict[int, list[Candidate]]:
    ordinary = retrieve_scene(root, split, scene, cache_root, workers=workers)
    expanded = retrieve_deep_scene(root, split, scene, cache_root, workers=workers)
    result: dict[int, list[Candidate]] = {}
    for number, base in ordinary.items():
        chosen = list(base)
        for candidate in expanded[number][:5]:
            matches = [index for index, current in enumerate(chosen)
                       if math.hypot(candidate.x-current.x,
                                     candidate.y-current.y) <= 8]
            if matches:
                index = matches[0]
                if candidate.confidence > chosen[index].confidence:
                    chosen[index] = candidate
            else:
                chosen.append(candidate)
        result[number] = sorted(chosen, key=lambda item: item.confidence,
                                reverse=True)[:15]
    return result
