"""Train and apply a candidate-level presence/localization model."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .matching import Candidate


FEATURE_NAMES = (
    "fine_score", "residual", "coarse_score", "gain", "blur",
    "outer_rmse", "outer_mae", "outer_corr", "background_rmse",
    "central_std", "target_std", "highpass_channel", "relative_rank",
    "ring_highpass_corr_1_5", "ring_highpass_corr_3",
)
UNION_FEATURE_NAMES = FEATURE_NAMES + ("legacy_source",)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64) - float(np.mean(first))
    second = second.astype(np.float64) - float(np.mean(second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-8 else 0.0


def feature_vector(image: np.ndarray, patch: np.ndarray,
                   candidate: Candidate, rank: int = 0) -> np.ndarray:
    x, y = round(candidate.x), round(candidate.y)
    source = image[max(0, y-16):y+16, max(0, x-16):x+16].astype(np.float32)
    if source.shape != (32, 32):
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    transform = cv2.getRotationMatrix2D((15.5, 15.5), candidate.angle, candidate.scale)
    target = cv2.warpAffine(patch, transform, (32, 32),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    valid = cv2.warpAffine(np.ones((32, 32), np.uint8), transform, (32, 32),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    yy, xx = np.mgrid[-16:16, -16:16]
    radius_squared = xx*xx + yy*yy
    inner = valid & (radius_squared < 100)
    outer = valid & (radius_squared >= 100) & (radius_squared < 245)
    if inner.sum() < 100 or outer.sum() < 100:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    design = np.stack((source[inner], np.ones(inner.sum())), axis=1)
    gain, offset = np.linalg.lstsq(design, target[inner], rcond=None)[0]
    predicted = source*gain + offset
    outer_error = target[outer] - predicted[outer]
    outer_rmse = float(np.sqrt(np.mean(outer_error**2)))
    outer_mae = float(np.mean(np.abs(outer_error)))
    outer_corr = _correlation(source[outer], target[outer])
    background = valid & (target < np.percentile(target[valid], 65))
    bg_error = target[background] - predicted[background]
    bg_rmse = float(np.sqrt(np.mean(bg_error**2)))
    ring_source = source[3:29, 3:29]
    ring_target = target[3:29, 3:29]
    ring_y, ring_x = np.mgrid[-13:13, -13:13]
    ring_radius_squared = ring_x*ring_x + ring_y*ring_y
    ring = (ring_radius_squared > 36) & (ring_radius_squared < 155)
    ring_correlations = []
    for sigma in (1.5, 3.0):
        source_highpass = ring_source - cv2.GaussianBlur(ring_source, (0, 0), sigma)
        target_highpass = ring_target - cv2.GaussianBlur(ring_target, (0, 0), sigma)
        a, b = source_highpass[ring], target_highpass[ring]
        ring_correlations.append(_correlation(a, b))
    return np.asarray((candidate.fine_score, candidate.residual,
                       candidate.coarse_score, candidate.gain, candidate.blur,
                       outer_rmse, outer_mae, outer_corr, bg_rmse,
                       float(source[inner].std()), float(target[inner].std()),
                       float(candidate.channel == "highpass"), float(rank),
                       *ring_correlations), dtype=np.float64)


class CandidateClassifier:
    def __init__(self, scaler: StandardScaler, model: LogisticRegression,
                 feature_names: tuple[str, ...] = FEATURE_NAMES) -> None:
        self.scaler = scaler
        self.model = model
        self.feature_names = feature_names

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray,
            feature_names: tuple[str, ...] = FEATURE_NAMES) -> "CandidateClassifier":
        if features.shape[1] != len(feature_names):
            raise ValueError("Feature matrix and feature names disagree")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        model = LogisticRegression(C=0.35, class_weight="balanced", max_iter=1000,
                                   random_state=6643)
        model.fit(scaled, labels)
        return cls(scaler, model, feature_names)

    def extract(self, image: np.ndarray, patch: np.ndarray,
                candidate: Candidate, rank: int = 0) -> np.ndarray:
        features = feature_vector(image, patch, candidate,
                                  0 if candidate.channel == "legacy" else rank)
        if self.feature_names == UNION_FEATURE_NAMES:
            return np.append(features, float(candidate.channel == "legacy"))
        return features

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "features": self.feature_names,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "coefficient": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CandidateClassifier":
        content = json.loads(path.read_text())
        feature_names = tuple(content["features"])
        if feature_names not in (FEATURE_NAMES, UNION_FEATURE_NAMES):
            raise ValueError("Incompatible candidate model features")
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(content["mean"])
        scaler.scale_ = np.asarray(content["scale"])
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(feature_names)
        model = LogisticRegression()
        model.classes_ = np.asarray([0, 1])
        model.coef_ = np.asarray([content["coefficient"]])
        model.intercept_ = np.asarray([content["intercept"]])
        model.n_features_in_ = len(feature_names)
        return cls(scaler, model, feature_names)
