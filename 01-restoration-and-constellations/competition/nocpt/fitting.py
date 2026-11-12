"""Bounded photometric verification of proposed patch locations."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .matching import Candidate, transformed_template


_AXIS = np.linspace(-1.0, 1.0, 26, dtype=np.float32)
_GRID_X, _GRID_Y = np.meshgrid(_AXIS, _AXIS)
_MASK = (_GRID_X**2 + _GRID_Y**2 < 0.94) & (_GRID_X**2 + _GRID_Y**2 > 0.04)


def verify_candidate(image: np.ndarray, patch: np.ndarray,
                     candidate: Candidate) -> Candidate:
    x, y = round(candidate.x), round(candidate.y)
    if x < 13 or y < 13 or x+13 > image.shape[1] or y+13 > image.shape[0]:
        return candidate
    source = image[y-13:y+13, x-13:x+13].astype(np.float32)
    target = transformed_template(patch, candidate.angle, candidate.scale).astype(np.float32)
    active = _MASK & (target > 4) & (target < 250)
    if active.sum() < 250 or target[active].std() < 3:
        return candidate
    sigma = max(6.0, float(target[active].std()))

    def prediction(parameters: np.ndarray) -> np.ndarray:
        gain, offset, grad_x, grad_y, blur = parameters
        blurred = cv2.GaussianBlur(source, (0, 0), max(0.01, float(blur))) if blur > 0.08 else source
        illumination = 1.0 + grad_x*_GRID_X + grad_y*_GRID_Y
        return gain * blurred * illumination + offset

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return ((prediction(parameters) - target) / sigma)[active].ravel()

    try:
        solution = least_squares(residuals, [1, 0, 0, 0, 0.45],
                                 bounds=([0.35, -120, -0.40, -0.40, 0.01],
                                         [2.6, 120, 0.40, 0.40, 3.0]),
                                 loss="soft_l1", f_scale=0.55, max_nfev=22)
    except (ValueError, cv2.error):
        return candidate
    normalized = residuals(solution.x)
    trimmed = np.sort(np.abs(normalized))[:int(0.90*len(normalized))]
    candidate.residual = float(np.sqrt(np.mean(trimmed**2)))
    candidate.gain = float(solution.x[0])
    candidate.blur = float(solution.x[4])
    boundary_penalty = 0.05 * max(0, solution.x[4]-2.0) + 0.08 * max(0, solution.x[0]-1.9)
    candidate.confidence = float(
        1.4*candidate.fine_score - 0.65*candidate.residual - boundary_penalty
    )
    return candidate
