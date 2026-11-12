"""Rotation/scale candidate retrieval and full-resolution refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Candidate:
    x: float
    y: float
    angle: float
    scale: float
    coarse_score: float
    channel: str
    fine_score: float = -1.0
    residual: float = 1.0
    gain: float = 0.0
    blur: float = 0.0
    confidence: float = 0.0

    def serializable(self) -> dict:
        return asdict(self)


def highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    array = image.astype(np.float32)
    smooth = cv2.GaussianBlur(array, (0, 0), sigma)
    return np.clip(array - smooth + 128, 0, 255).astype(np.uint8)


def transformed_template(patch: np.ndarray, angle: float, scale: float,
                         margin: int = 3) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((15.5, 15.5), angle, scale)
    rotated = cv2.warpAffine(patch, matrix, (32, 32), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return rotated[margin:32-margin, margin:32-margin]


def _top_peaks(response: np.ndarray, count: int, radius: int) -> Iterable[tuple[int, int, float]]:
    for _ in range(count):
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        yield x, y, score
        response[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -2


def _deduplicate(candidates: list[Candidate], limit: int, radius: float = 12.0) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.coarse_score, reverse=True):
        if all(math.hypot(candidate.x-other.x, candidate.y-other.y) > radius
               for other in selected):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def coarse_search(image_half: np.ndarray, patch: np.ndarray, angles: int = 36,
                  scales: tuple[float, ...] = (0.80, 0.93, 1.08, 1.24),
                  channel: str = "raw", keep: int = 12) -> list[Candidate]:
    scene = image_half if channel == "raw" else highpass(image_half, 2.0)
    candidates: list[Candidate] = []
    patch_half = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
    for scale in scales:
        for angle in np.linspace(0, 360, angles, endpoint=False):
            template = transformed_template(patch, float(angle), scale)
            template = cv2.resize(template, (13, 13), interpolation=cv2.INTER_AREA)
            if channel != "raw":
                template = highpass(template, 1.8)
            if template.std() < 2.5:
                continue
            response = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
            for x, y, score in _top_peaks(response, 2, 7):
                if not np.isfinite(score):
                    continue
                # The 26-pixel centre crop becomes 13 pixels at half resolution.
                candidates.append(Candidate(2*x+13, 2*y+13, float(angle), scale,
                                            float(score), channel))
    return _deduplicate(candidates, keep)


def fine_search(image: np.ndarray, patch: np.ndarray,
                candidates: list[Candidate], keep: int = 8) -> list[Candidate]:
    refined: list[Candidate] = []
    for candidate in candidates:
        best = None
        for angle in (candidate.angle-5, candidate.angle, candidate.angle+5):
            for scale in (candidate.scale*0.95, candidate.scale, candidate.scale*1.05):
                template = transformed_template(patch, angle, scale, margin=3)
                if candidate.channel != "raw":
                    template = highpass(template, 3.0)
                centre_x, centre_y = round(candidate.x), round(candidate.y)
                x0 = max(0, centre_x-19)
                y0 = max(0, centre_y-19)
                x1 = min(image.shape[1], centre_x+20)
                y1 = min(image.shape[0], centre_y+20)
                source = image[y0:y1, x0:x1]
                if source.shape[0] < 26 or source.shape[1] < 26:
                    continue
                if candidate.channel != "raw":
                    source = highpass(source, 3.0)
                response = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                if best is None or score > best[0]:
                    best = score, x0+x+13, y0+y+13, angle, scale
        if best:
            candidate.fine_score = float(best[0])
            candidate.x = float(best[1])
            candidate.y = float(best[2])
            candidate.angle = float(best[3] % 360)
            candidate.scale = float(best[4])
            refined.append(candidate)
    refined.sort(key=lambda c: c.fine_score, reverse=True)
    kept: list[Candidate] = []
    for candidate in refined:
        if all(math.hypot(candidate.x - other.x, candidate.y - other.y) > 8
               for other in kept):
            kept.append(candidate)
            if len(kept) == keep:
                break
    return kept
