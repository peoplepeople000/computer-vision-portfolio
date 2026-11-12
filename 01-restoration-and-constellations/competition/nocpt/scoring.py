"""Local implementation of the published competition metric."""

from __future__ import annotations

import math
from .data import parse_cell


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _reward(distance: float) -> float:
    return 1.0 if distance <= 12 else max(0.0, (36 - distance) / 24)


def score_scene(truth: dict[str, str], prediction: dict[str, str]) -> dict[str, float]:
    n = int(truth["n_patches"])
    actual = [parse_cell(truth[f"patch_{i:02d}"]) for i in range(1, n + 1)]
    predicted = [parse_cell(prediction[f"patch_{i:02d}"]) for i in range(1, n + 1)]
    tp = sum(a is not None and p is not None for a, p in zip(actual, predicted))
    fp = sum(a is None and p is not None for a, p in zip(actual, predicted))
    fn = sum(a is not None and p is None for a, p in zip(actual, predicted))
    tn = n - tp - fp - fn
    presence = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
    true_present = [(a, p) for a, p in zip(actual, predicted) if a is not None]
    localization = sum(
        _reward(math.dist(a[:2], p[:2])) if p is not None else 0.0
        for a, p in true_present
    ) / max(1, len(true_present))
    figure = [a for a in actual if a is not None and a[2] == 1]
    reported = [p for p in predicted if p is not None]
    pairs = sorted(
        (math.dist(a[:2], p[:2]), i, j)
        for i, a in enumerate(figure)
        for j, p in enumerate(reported)
    )
    used_true: set[int] = set()
    used_pred: set[int] = set()
    recovery = 0.0
    for distance, i, j in pairs:
        if distance >= 36:
            break
        if i not in used_true and j not in used_pred:
            recovery += _reward(distance)
            used_true.add(i)
            used_pred.add(j)
    recovery /= max(1, len(figure))
    identification = float(truth["constellation"] == prediction["constellation"])
    total = 0.25 * presence + 0.20 * localization + 0.25 * recovery + 0.30 * identification
    return {"presence": presence, "localization": localization,
            "geometric_recovery": recovery, "identification": identification,
            "score": total}
