"""Quadratic subpixel localization around a full-resolution NCC peak."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .matching import Candidate, highpass, transformed_template


def _vertex(left: float, center: float, right: float) -> float:
    curvature = left - 2*center + right
    if curvature >= -1e-8:
        return 0.0
    return float(np.clip(0.5*(left-right)/curvature, -0.5, 0.5))


def refine_center(image: np.ndarray, patch: np.ndarray,
                  candidate: Candidate) -> Candidate:
    x, y = round(candidate.x), round(candidate.y)
    source = image[y-15:y+15, x-15:x+15]
    if source.shape != (30, 30):
        return candidate
    template = transformed_template(patch, candidate.angle, candidate.scale)
    if candidate.channel != "raw":
        source = highpass(source, 3.0)
        template = highpass(template, 3.0)
    response = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, (peak_x, peak_y) = cv2.minMaxLoc(response)
    if not (0 < peak_x < response.shape[1]-1 and
            0 < peak_y < response.shape[0]-1):
        return candidate
    offset_x = peak_x-2 + _vertex(float(response[peak_y, peak_x-1]),
                                  float(response[peak_y, peak_x]),
                                  float(response[peak_y, peak_x+1]))
    offset_y = peak_y-2 + _vertex(float(response[peak_y-1, peak_x]),
                                  float(response[peak_y, peak_x]),
                                  float(response[peak_y+1, peak_x]))
    if abs(offset_x) > 1.5 or abs(offset_y) > 1.5:
        return candidate
    return replace(candidate, x=x+offset_x, y=y+offset_y)
