"""Search for a reference constellation among noisy recovered patch centres."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .matching import Candidate


@dataclass
class PatternResult:
    name: str
    score: float
    matched: int
    nodes: int
    matrix: np.ndarray
    projected: np.ndarray

    def serializable(self) -> dict:
        return {"name": self.name, "score": self.score, "matched": self.matched,
                "nodes": self.nodes, "matrix": self.matrix.tolist(),
                "projected": self.projected.tolist()}


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _evaluate(nodes: np.ndarray, matrix: np.ndarray, points: np.ndarray,
              weights: np.ndarray, tree: cKDTree, image: np.ndarray | None,
              radius: float = 28.0, saturation: np.ndarray | None = None,
              similarity: np.ndarray | None = None) -> tuple[float, int, np.ndarray]:
    projected = _transform(nodes, matrix)
    visible = np.all((projected >= 5) & (projected < 2995), axis=1)
    if visible.sum() < max(3, math.ceil(0.60*len(nodes))):
        return -100.0, 0, projected
    distance, indices = tree.query(projected, distance_upper_bound=radius)
    valid = (distance < radius) & visible
    used: set[int] = set()
    matched_indices: list[int] = []
    matched = 0
    support = 0.0
    for index in np.argsort(distance):
        if valid[index] and int(indices[index]) not in used:
            used.add(int(indices[index]))
            matched_indices.append(int(indices[index]))
            matched += 1
            support += weights[indices[index]] * (1.0 - 0.35 * distance[index] / radius)
    saturated = 0
    if saturation is not None:
        positions = np.rint(projected[visible]).astype(int)
        saturated = int(saturation[positions[:, 1], positions[:, 0]].sum())
    # A two-point diagram can be matched by almost any two detected stars.
    complexity_penalty = 3.0 if len(nodes) <= 2 else (1.2 if len(nodes) == 3 else 0.0)
    coherence = 0.0
    if similarity is not None and len(matched_indices) >= 3:
        submatrix = similarity[np.ix_(matched_indices, matched_indices)]
        coherence = float((submatrix.sum() - len(matched_indices)) /
                          (len(matched_indices)*(len(matched_indices)-1)))
    score = (support + 0.45 * matched + 0.85 * matched / visible.sum()
             + 0.9 * saturated - 0.5 * (visible.sum()-saturated)
             - 0.30 * (len(nodes)-visible.sum())
             - 0.08 * len(nodes) - complexity_penalty
             + 5.0 * coherence * min(1.0, matched/5))
    return float(score), matched, projected


def _affine_refinement(nodes: np.ndarray, matrix: np.ndarray, points: np.ndarray,
                       tree: cKDTree, radius: float = 85.0) -> np.ndarray | None:
    transformed = _transform(nodes, matrix)
    distances, indices = tree.query(transformed, distance_upper_bound=radius)
    matches = [(i, int(indices[i])) for i in np.argsort(distances)
               if distances[i] < radius]
    used: set[int] = set()
    filtered = []
    for i, j in matches:
        if j not in used:
            filtered.append((i, j))
            used.add(j)
    if len(filtered) < 3:
        return None
    source = nodes[[i for i, _ in filtered]]
    target = points[[j for _, j in filtered]]
    design = np.column_stack((source, np.ones(len(source))))
    refined = np.linalg.lstsq(design, target, rcond=None)[0].T
    singular = np.linalg.svd(refined[:, :2], compute_uv=False)
    if singular[1] < 0.2 or singular[0] / singular[1] > 2.5:
        return None
    initial_scale = np.linalg.norm(matrix[:, :2], axis=0).mean()
    if not 0.45 < singular.mean() / initial_scale < 2.2:
        return None
    return refined


def rank_patterns(patterns: dict[str, np.ndarray], candidates: dict[int, list[Candidate]],
                  image: np.ndarray | None = None,
                  min_evidence: float = 0.80) -> list[PatternResult]:
    def evidence(candidate: Candidate) -> float:
        if candidate.confidence > 0:
            return candidate.confidence
        # Saturation can leave too few usable pixels for the photometric fit.
        return 1.4*candidate.fine_score - 0.5 if candidate.fine_score >= 0.95 else 0.0

    chosen = [(index, choices[0]) for index, choices in candidates.items()
              if choices and evidence(choices[0]) >= min_evidence]
    if len(chosen) < 3:
        return []
    points = np.array([[candidate.x, candidate.y] for _, candidate in chosen], dtype=np.float64)
    weights = np.array([np.clip(evidence(candidate), 0.3, 1.0)
                        for _, candidate in chosen], dtype=np.float64)
    tree = cKDTree(points)
    saturation = None
    similarity = None
    if image is not None:
        saturation = cv2.dilate((image >= 252).astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        profiles = []
        for x, y in points:
            xx, yy = round(x), round(y)
            if 7 <= xx < image.shape[1]-7 and 7 <= yy < image.shape[0]-7:
                crop = image[yy-7:yy+8, xx-7:xx+8].astype(np.float32)
                crop -= cv2.GaussianBlur(crop, (0, 0), 3.0)
                vector = crop.ravel()
                vector -= vector.mean()
                vector /= max(1e-6, float(np.linalg.norm(vector)))
            else:
                vector = np.zeros(225, dtype=np.float32)
            profiles.append(vector)
        profile_matrix = np.stack(profiles)
        similarity = profile_matrix @ profile_matrix.T
    results: list[PatternResult] = []
    for name, nodes in patterns.items():
        n = len(nodes)
        if n < 3:
            continue
        best: list[tuple[float, np.ndarray]] = []
        for mirror in (False, True):
            source = nodes.copy()
            if mirror:
                source[:, 0] *= -1
            source_i, source_j = np.where(~np.eye(n, dtype=bool))
            target_i, target_j = np.where(~np.eye(len(points), dtype=bool))
            target_delta = points[target_j] - points[target_i]
            for start in range(0, len(source_i), 20):
                source_a = source_i[start:start+20]
                source_b = source_j[start:start+20]
                i = np.repeat(source_a, len(target_i))
                j = np.repeat(source_b, len(target_i))
                a = np.tile(target_i, len(source_a))
                src_delta = source[j] - source[i]
                dst_delta = np.tile(target_delta, (len(source_a), 1))
                denom = np.sum(src_delta**2, axis=1)
                real = np.sum(dst_delta*src_delta, axis=1)/denom
                imag = (dst_delta[:, 1]*src_delta[:, 0] - dst_delta[:, 0]*src_delta[:, 1])/denom
                scale = np.hypot(real, imag)
                valid = (scale > 0.35) & (scale < 16)
                if not np.any(valid):
                    continue
                real, imag, src_i, dst_a = real[valid], imag[valid], source[i[valid]], points[a[valid]]
                matrix = np.stack((np.stack((real, -imag), axis=-1),
                                   np.stack((imag, real), axis=-1)), axis=1)
                offset = dst_a - np.einsum("bij,bj->bi", matrix, src_i)
                projection = np.einsum("bij,nj->bni", matrix, source) + offset[:, None, :]
                visible = np.all((projection >= 5) & (projection < 2995), axis=2)
                bounds = visible.sum(axis=1) >= max(3, math.ceil(0.60*n))
                if not np.any(bounds):
                    continue
                projection = projection[bounds]
                matrix, offset = matrix[bounds], offset[bounds]
                distances, indices = tree.query(projection.reshape(-1, 2),
                                                distance_upper_bound=42)
                distances = distances.reshape(-1, n)
                indices = indices.reshape(-1, n)
                hits = distances < 42
                # Deduplicate target points within each candidate transform.
                sorted_indices = np.sort(np.where(hits, indices, len(points)), axis=1)
                unique_hits = np.sum((sorted_indices[:, 1:] != sorted_indices[:, :-1])
                                     & (sorted_indices[:, 1:] < len(points)), axis=1)
                unique_hits += (sorted_indices[:, 0] < len(points)).astype(int)
                primary = unique_hits + 0.15*np.sum(hits, axis=1)
                if saturation is not None:
                    pixel = np.clip(np.rint(projection).astype(np.int32), 0, 2999)
                    visible = np.all((projection >= 5) & (projection < 2995), axis=2)
                    saturations = (saturation[pixel[:, :, 1], pixel[:, :, 0]] * visible).sum(axis=1)
                    primary += 0.8*saturations
                top = np.argpartition(primary, -min(5, len(primary)))[-min(5, len(primary)):]
                for row in top:
                    if unique_hits[row] < 2:
                        continue
                    linear = matrix[row].copy()
                    if mirror:
                        linear[:, 0] *= -1
                    transform = np.column_stack((linear, offset[row]))
                    score, _, _ = _evaluate(nodes, transform, points, weights, tree,
                                            image, radius=42, saturation=saturation,
                                            similarity=similarity)
                    best.append((score, transform))
                best = sorted(best, key=lambda item: item[0], reverse=True)[:50]
        best_result: PatternResult | None = None
        for _, matrix in best:
            alternatives = [matrix]
            refined = _affine_refinement(nodes, matrix, points, tree)
            if refined is not None:
                alternatives.append(refined)
            for candidate_matrix in alternatives:
                score, matched, projected = _evaluate(
                    nodes, candidate_matrix, points, weights, tree, image,
                    saturation=saturation, similarity=similarity)
                if refined is not None and np.array_equal(candidate_matrix, refined):
                    singular = np.linalg.svd(refined[:, :2], compute_uv=False)
                    score -= 0.15 * (singular[0]/singular[1]-1)
                if best_result is None or score > best_result.score:
                    best_result = PatternResult(name, score, matched, len(nodes),
                                                candidate_matrix, projected)
        if best_result:
            results.append(best_result)
    return sorted(results, key=lambda result: result.score, reverse=True)


def assign_membership(candidates: dict[int, list[Candidate]],
                      pattern: PatternResult | None, distance: float = 24) -> dict[int, bool]:
    if pattern is None:
        return {index: False for index in candidates}
    tree = cKDTree(pattern.projected)
    result = {}
    for index, choices in candidates.items():
        result[index] = bool(choices and tree.query([choices[0].x, choices[0].y])[0] < distance)
    return result
