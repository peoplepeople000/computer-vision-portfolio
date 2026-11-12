"""Compare a sky location with a query in the query's own coordinate frame."""

from __future__ import annotations

import cv2
import numpy as np

from .matching import Candidate


_U, _V = np.meshgrid(np.arange(32, dtype=np.float32) - 15.5,
                     np.arange(32, dtype=np.float32) - 15.5)
_R2 = _U*_U + _V*_V
_CORE = _R2 < 64
_RING = (_R2 >= 64) & (_R2 < 220)
_BORDER = _R2 >= 144

EVIDENCE_NAMES = (
    "aligned_full_corr", "aligned_core_corr", "aligned_ring_corr",
    "aligned_border_corr", "aligned_ring_hp_1_5", "aligned_ring_hp_3",
    "aligned_trimmed_error", "aligned_ring_error", "aligned_best_blur",
    "aligned_best_blur_corr", "aligned_best_blur_error",
)


def _corr(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = first[mask].astype(np.float64)
    b = second[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a, b)/denominator) if denominator > 1e-8 else 0.0


def _fit_error(source: np.ndarray, target: np.ndarray,
               fit_mask: np.ndarray, measure_mask: np.ndarray,
               trimmed: bool = False) -> float:
    design = np.column_stack((source[fit_mask],
                              np.ones(int(fit_mask.sum()), dtype=np.float32)))
    gain, offset = np.linalg.lstsq(design, target[fit_mask], rcond=None)[0]
    error = np.abs(target[measure_mask] - (gain*source[measure_mask] + offset))
    if trimmed:
        error = np.sort(error)[:max(1, int(0.85*len(error)))]
    scale = max(8.0, float(target[measure_mask].std()))
    return float(np.sqrt(np.mean(error*error))/scale)


def evidence_vector(image: np.ndarray, patch: np.ndarray,
                    candidate: Candidate) -> np.ndarray:
    """Return fixed-length evidence without fitting to a scene label."""
    angle = np.deg2rad(candidate.angle)
    real = np.float32(candidate.scale*np.cos(angle))
    imag = np.float32(candidate.scale*np.sin(angle))
    map_x = np.asarray(candidate.x + real*_U + imag*_V, dtype=np.float32)
    map_y = np.asarray(candidate.y - imag*_U + real*_V, dtype=np.float32)
    source = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    target = patch.astype(np.float32)
    valid = (target > 4) & (target < 250)
    ring = _RING & valid
    border = _BORDER & valid
    full = valid
    if full.sum() < 300 or ring.sum() < 100 or border.sum() < 100:
        return np.zeros(len(EVIDENCE_NAMES), dtype=np.float64)
    core = _CORE & valid
    scores = [
        _corr(source, target, full),
        _corr(source, target, core) if core.sum() > 25 else 0.0,
        _corr(source, target, ring),
        _corr(source, target, border),
    ]
    for sigma in (1.5, 3.0):
        source_hp = source - cv2.GaussianBlur(source, (0, 0), sigma)
        target_hp = target - cv2.GaussianBlur(target, (0, 0), sigma)
        scores.append(_corr(source_hp, target_hp, ring))
    scores.extend((
        _fit_error(source, target, ring, full, trimmed=True),
        _fit_error(source, target, ring, ring),
    ))
    blur_scores = []
    for sigma in (0.8, 1.6, 2.6):
        blurred = cv2.GaussianBlur(source, (0, 0), sigma)
        blur_scores.append((_fit_error(blurred, target, ring, full, trimmed=True),
                            sigma, _corr(blurred, target, ring)))
    best_error, best_blur, best_corr = min(blur_scores)
    scores.extend((best_blur, best_corr, best_error))
    return np.asarray(scores, dtype=np.float64)
