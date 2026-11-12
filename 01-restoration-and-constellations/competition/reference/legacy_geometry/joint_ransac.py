"""Candidate-aware constellation fitting without ground-truth inputs.

The matcher expects one scene at a time.  Candidate coordinates are panorama
coordinates for *reported* query locations; ``geom_x``/``geom_y`` may be used
when the star center differs from the coordinate reported in a submission.
Off-figure queries are permitted: a fitted constellation only selects the
queries that support its nodes, and unmatched baseline predictions survive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class JointConfig:
    seed: int = 6643
    proposals_per_pattern: int = 2400
    max_candidates_per_patch: int = 5
    keep_proposals: int = 12
    min_consensus: int = 3
    inlier_radius: float = 18.0
    gaussian_sigma: float = 6.0
    min_scale: float = 0.25
    max_scale: float = 40.0
    min_node_pair_fraction: float = 0.12
    max_affine_anisotropy: float = 2.5
    min_affine_support: int = 5
    allow_affine: bool = True
    max_refinement_rounds: int = 3
    candidate_temperature: float = 0.15
    node_count_penalty: float = 0.015
    baseline_member_seed_boost: float = 3.0
    triangle_seeds_per_pose: int = 12
    null_shift_weight: float = 0.5
    null_shift_radius_multiplier: float = 3.0


@dataclass
class JointResult:
    rankings: pd.DataFrame
    selections: pd.DataFrame
    best_pose: dict | None


def _check_points(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
        raise ValueError("Each pattern must contain at least two 2D nodes")
    if not np.isfinite(p).all():
        raise ValueError("Pattern nodes must be finite")
    if np.unique(p, axis=0).shape[0] != p.shape[0]:
        raise ValueError("Pattern nodes must have unique coordinates")
    return p


def _prepare(
    candidates: pd.DataFrame, baseline: pd.DataFrame | None, config: JointConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"patch_id", "x", "y"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate table needs {sorted(required)}")
    score_column = "appearance_score" if "appearance_score" in candidates else "score"
    if score_column not in candidates:
        raise ValueError("Candidate table needs appearance_score or score")
    if candidates.empty:
        raise ValueError("Candidate table is empty")
    c = candidates.copy()
    c["appearance_score"] = pd.to_numeric(c[score_column], errors="raise")
    c["geom_x"] = pd.to_numeric(c.get("geom_x", c["x"]), errors="raise")
    c["geom_y"] = pd.to_numeric(c.get("geom_y", c["y"]), errors="raise")
    c = c.sort_values(
        ["patch_id", "appearance_score", "x", "y"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    if "source" in c:
        # The frozen v6 location is a genuine candidate even when a large
        # retrieved bank gives it a low appearance rank. Preserve one saved
        # candidate per query so the joint fit can choose the old location.
        groups = []
        for _, group in c.groupby("patch_id", sort=False):
            chosen = group.head(config.max_candidates_per_patch)
            protected = group[group["source"].ne("retrieved")]
            if not protected.empty:
                frozen = protected[protected["source"].eq("v6_submission")]
                saved = frozen.iloc[[0]] if not frozen.empty else protected.iloc[[0]]
                if not chosen.index.isin(saved.index).any():
                    chosen = pd.concat([chosen.iloc[:-1], saved])
            groups.append(chosen)
        c = pd.concat(groups).sort_values(
            ["patch_id", "appearance_score", "x", "y"],
            ascending=[True, False, True, True], kind="stable",
        )
    else:
        c = c.groupby("patch_id", sort=False).head(config.max_candidates_per_patch)
    c = c.reset_index(drop=True)
    values = c[["x", "y", "geom_x", "geom_y", "appearance_score"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Candidate coordinates and scores must be finite")
    per_query_max = c.groupby("patch_id")["appearance_score"].transform("max")
    relative = np.exp(np.clip(
        (c["appearance_score"] - per_query_max) / config.candidate_temperature,
        -12.0, 0.0,
    ))
    c["_weight"] = 0.25 + 0.75 * relative
    if "presence_prob" in c:
        probability = pd.to_numeric(c["presence_prob"], errors="raise").clip(0, 1)
        c["_weight"] *= 0.5 + 0.5 * probability
    c["_row_id"] = np.arange(len(c), dtype=np.int32)

    if baseline is None:
        b = pd.DataFrame({"patch_id": c["patch_id"].drop_duplicates()})
        b["x"] = np.nan
        b["y"] = np.nan
    else:
        b = baseline.copy()
        if "patch_id" not in b or b["patch_id"].duplicated().any():
            raise ValueError("Baseline must have one row per patch_id")
        for coordinate in ("x", "y"):
            if coordinate not in b:
                b[coordinate] = np.nan
    return c, b


def _pair_pose(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray,
               reflection: bool) -> tuple[np.ndarray, np.ndarray] | None:
    source = b - a
    target = y - x
    ns = float(np.linalg.norm(source))
    nt = float(np.linalg.norm(target))
    if ns < 1e-8 or nt < 1e-8:
        return None
    flip = np.diag([-1.0, 1.0]) if reflection else np.eye(2)
    fs = flip @ source
    cosine = float(np.dot(fs, target) / (ns * nt))
    sine = float(np.linalg.det(np.stack([fs, target])) / (ns * nt))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    matrix = nt / ns * rotation @ flip
    return matrix, x - matrix @ a


def _score_pose(
    matrix: np.ndarray,
    translation: np.ndarray,
    nodes: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    patch_index: np.ndarray,
    tree: cKDTree,
    config: JointConfig,
    required_support: int | None = None,
    seed_deduction: int = 2,
    exact_assignment: bool = False,
) -> tuple[float, list[tuple[int, int, float, float]]]:
    mapped = nodes @ matrix.T + translation
    neighborhoods = tree.query_ball_point(mapped, r=config.inlier_radius)
    edges: list[tuple[float, int, int, float]] = []
    inv_two_sigma_sq = 0.5 / (config.gaussian_sigma ** 2)
    for node_index, rows in enumerate(neighborhoods):
        if not rows:
            continue
        row_array = np.asarray(rows, dtype=np.int32)
        distances = np.linalg.norm(positions[row_array] - mapped[node_index], axis=1)
        strength = weights[row_array] * np.exp(-(distances ** 2) * inv_two_sigma_sq)
        # Bounding the edge list matters when many star candidates cluster.
        best = np.argsort(strength)[-min(12, len(strength)):]
        edges.extend((float(strength[k]), int(row_array[k]), node_index,
                      float(distances[k])) for k in best)
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    matches = _assign_edges(edges, patch_index, len(nodes),
                            int(patch_index.max()) + 1, exact_assignment)
    required = config.min_consensus if required_support is None else required_support
    if len(matches) < required:
        return -np.inf, matches
    strengths = sorted((match[2] for match in matches), reverse=True)
    # Seed correspondences always fit exactly. Their score is removed.
    score = (sum(strengths[seed_deduction:])
             + 0.08 * (len(matches) - seed_deduction)
             - config.node_count_penalty * len(nodes))
    return float(score), matches


def _assign_edges(
    edges: list[tuple[float, int, int, float]],
    patch_index: np.ndarray,
    node_count: int,
    patch_count: int,
    exact: bool,
) -> list[tuple[int, int, float, float]]:
    if exact:
        # Columns after patch_count represent unmatched template nodes.
        benefits = np.zeros((node_count, patch_count + node_count))
        chosen: dict[tuple[int, int], tuple[int, int, float, float]] = {}
        for strength, row_id, node_id, residual in edges:
            patch_id = int(patch_index[row_id])
            if strength > benefits[node_id, patch_id]:
                benefits[node_id, patch_id] = strength
                chosen[(node_id, patch_id)] = (
                    row_id, node_id, strength, residual,
                )
        node_ids, patch_ids = linear_sum_assignment(-benefits)
        matches = [chosen[(int(n), int(p))]
                   for n, p in zip(node_ids, patch_ids)
                   if p < patch_count and benefits[n, p] > 0]
        matches.sort(key=lambda item: item[2], reverse=True)
        return matches
    used_patches: set[int] = set()
    used_nodes: set[int] = set()
    matches = []
    for strength, row_id, node_id, residual in edges:
        patch_id = int(patch_index[row_id])
        if patch_id in used_patches or node_id in used_nodes:
            continue
        used_patches.add(patch_id)
        used_nodes.add(node_id)
        matches.append((row_id, node_id, strength, residual))
    return matches


def _similarity_refit(source: np.ndarray, target: np.ndarray,
                      desired_reflection: bool) -> tuple[np.ndarray, np.ndarray] | None:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    denominator = float(np.sum(x * x))
    if denominator < 1e-8:
        return None
    u, _, vh = np.linalg.svd(x.T @ y)
    sign = -1.0 if desired_reflection else 1.0
    correction = np.diag([1.0, sign * np.linalg.det(u @ vh)])
    row_rotation = u @ correction @ vh
    scale = float(np.sum((x @ row_rotation) * y) / denominator)
    if scale <= 0:
        return None
    matrix = scale * row_rotation.T
    return matrix, target_mean - matrix @ source_mean


def _affine_refit(source: np.ndarray, target: np.ndarray,
                  config: JointConfig) -> tuple[np.ndarray, np.ndarray] | None:
    if len(source) < config.min_affine_support:
        return None
    centered = source - source.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[1] < 0.04 * singular[0]:
        return None
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = coefficients[:2].T
    values = np.linalg.svd(matrix, compute_uv=False)
    if values[-1] < 1e-8 or values[0] / values[-1] > config.max_affine_anisotropy:
        return None
    return matrix, coefficients[2]


def _triangle_affine(source: np.ndarray, target: np.ndarray,
                     config: JointConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact three-point hypothesis, admitted only after a fourth query agrees."""
    if source.shape != (3, 2) or target.shape != (3, 2):
        return None
    for triangle in (source, target):
        singular = np.linalg.svd(triangle - triangle.mean(axis=0),
                                 compute_uv=False)
        if singular[0] < 1e-8 or singular[1] < 0.06 * singular[0]:
            return None
    design = np.column_stack([source, np.ones(3)])
    coefficients = np.linalg.solve(design, target)
    matrix = coefficients[:2].T
    values = np.linalg.svd(matrix, compute_uv=False)
    if (values[-1] < config.min_scale or values[0] > config.max_scale or
            values[0] / values[-1] > config.max_affine_anisotropy):
        return None
    return matrix, coefficients[2]


def _triangle_hypotheses(
    top: list[tuple[float, np.ndarray, np.ndarray, list]],
    nodes: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    patch_index: np.ndarray,
    tree: cKDTree,
    config: JointConfig,
) -> list[tuple[float, np.ndarray, np.ndarray, list]]:
    if len(nodes) < 4 or config.triangle_seeds_per_pose <= 0:
        return []
    proposed = []
    for _, _, _, matches in top:
        # Strong matched queries from a pair fit define plausible asterisms.
        for indices in list(combinations(range(min(len(matches), 6)), 3))[
                :config.triangle_seeds_per_pose]:
            source = np.stack([nodes[matches[i][1]] for i in indices])
            target = np.stack([positions[matches[i][0]] for i in indices])
            fitted = _triangle_affine(source, target, config)
            if fitted is None:
                continue
            matrix, translation = fitted
            # Three exact matches are tautological; a fourth independent
            # query is needed before this affine pose enters the ranking.
            heldout_score, matched = _score_pose(
                matrix, translation, nodes, positions, weights,
                patch_index, tree, config,
                required_support=max(4, config.min_consensus + 1),
                seed_deduction=3,
            )
            if np.isfinite(heldout_score):
                comparable_score, matched = _score_pose(
                    matrix, translation, nodes, positions, weights,
                    patch_index, tree, config,
                )
                proposed.append((comparable_score, matrix, translation, matched))
    proposed.sort(key=lambda item: item[0], reverse=True)
    return proposed[:config.keep_proposals]


def _shifted_null_mean(
    matrix: np.ndarray,
    translation: np.ndarray,
    nodes: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    patch_index: np.ndarray,
    tree: cKDTree,
    config: JointConfig,
) -> float:
    radius = config.null_shift_radius_multiplier * config.inlier_radius
    shifts = np.array([[radius, 0], [-radius, 0], [0, radius], [0, -radius],
                       [radius, radius], [-radius, -radius]])
    scores = []
    for shift in shifts:
        value, _ = _score_pose(matrix, translation + shift, nodes, positions,
                               weights, patch_index, tree, config,
                               exact_assignment=True)
        scores.append(max(0.0, value) if np.isfinite(value) else 0.0)
    return float(np.mean(scores))


def _refine(
    hypothesis: tuple[float, np.ndarray, np.ndarray, list],
    nodes: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    patch_index: np.ndarray,
    tree: cKDTree,
    config: JointConfig,
) -> tuple[float, np.ndarray, np.ndarray, list]:
    best = hypothesis
    for _ in range(config.max_refinement_rounds):
        score, matrix, translation, matches = best
        source = np.stack([nodes[match[1]] for match in matches])
        target = np.stack([positions[match[0]] for match in matches])
        proposals = [_similarity_refit(source, target, np.linalg.det(matrix) < 0)]
        if config.allow_affine:
            proposals.append(_affine_refit(source, target, config))
        improved = False
        for fitted in proposals:
            if fitted is None:
                continue
            new_matrix, new_translation = fitted
            singular = np.linalg.svd(new_matrix, compute_uv=False)
            if singular[-1] < config.min_scale or singular[0] > config.max_scale:
                continue
            new_score, new_matches = _score_pose(
                new_matrix, new_translation, nodes, positions, weights,
                patch_index, tree, config, exact_assignment=True,
            )
            if new_score > best[0] + 1e-7:
                best = (new_score, new_matrix, new_translation, new_matches)
                improved = True
        if not improved:
            break
    return best


def fit_joint_constellations(
    candidates: pd.DataFrame,
    pattern_nodes: Mapping[str, np.ndarray],
    baseline: pd.DataFrame | None = None,
    config: JointConfig = JointConfig(),
) -> JointResult:
    """Rank pattern poses and select one geometrically consistent candidate/query.

    Only appearance, candidate geometry, pattern nodes, and optional baseline
    *predictions* are used. The baseline supplies a seed prior when it contains
    ``m``/``member``; it never supplies ground truth. A match does not change a
    query's presence decision: evaluate that separately before CSV export.
    """
    if not pattern_nodes:
        raise ValueError("No patterns supplied")
    if config.min_consensus < 3 or config.proposals_per_pattern < 1:
        raise ValueError("Use at least three-query consensus and one proposal")
    c, b = _prepare(candidates, baseline, config)
    patch_ids = c["patch_id"].drop_duplicates().tolist()
    patch_lookup = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    patch_index = c["patch_id"].map(patch_lookup).to_numpy(dtype=np.int32)
    positions = c[["geom_x", "geom_y"]].to_numpy(dtype=float)
    weights = c["_weight"].to_numpy(dtype=float)
    tree = cKDTree(positions)
    rows_by_patch = [np.flatnonzero(patch_index == i) for i in range(len(patch_ids))]
    row_probs = [weights[rows] / weights[rows].sum() for rows in rows_by_patch]
    patch_probs = np.ones(len(patch_ids), dtype=float)
    membership_col = "m" if "m" in b else "member" if "member" in b else None
    if membership_col:
        baseline_member = b.set_index("patch_id")[membership_col].to_dict()
        for i, patch_id in enumerate(patch_ids):
            if baseline_member.get(patch_id) == 1:
                patch_probs[i] *= config.baseline_member_seed_boost
    patch_probs /= patch_probs.sum()
    records: list[dict] = []
    winners: dict[str, tuple[float, np.ndarray, np.ndarray, list]] = {}
    winner_nulls: dict[str, float] = {}

    for name, raw_nodes in sorted(pattern_nodes.items()):
        nodes = _check_points(raw_nodes)
        two_node_fallback = len(nodes) == 2
        if (len(nodes) < 2 or len(patch_ids) <
                (2 if two_node_fallback else config.min_consensus)):
            records.append({"pattern": name, "score": -np.inf, "support": 0,
                            "status": "insufficient_nodes_or_queries"})
            continue
        distances = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=2)
        diameter = float(distances.max())
        pairs = np.argwhere(np.triu(
            distances >= config.min_node_pair_fraction * diameter, 1
        ))
        if len(pairs) == 0:
            raise ValueError(f"Pattern {name} has no nondegenerate node pair")
        # A stable name-derived seed keeps results independent of dict order.
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        rng = np.random.default_rng(config.seed ^ name_seed)
        top: list[tuple[float, np.ndarray, np.ndarray, list]] = []
        for _ in range(config.proposals_per_pattern):
            first_patch = int(rng.choice(len(patch_ids), p=patch_probs))
            second_probs = patch_probs.copy()
            second_probs[first_patch] = 0.0
            second_probs /= second_probs.sum()
            second_patch = int(rng.choice(len(patch_ids), p=second_probs))
            first_row = int(rng.choice(rows_by_patch[first_patch], p=row_probs[first_patch]))
            second_row = int(rng.choice(rows_by_patch[second_patch], p=row_probs[second_patch]))
            pair = pairs[int(rng.integers(len(pairs)))]
            node_a, node_b = int(pair[0]), int(pair[1])
            if rng.integers(2):
                node_a, node_b = node_b, node_a
            fitted = _pair_pose(nodes[node_a], nodes[node_b],
                                positions[first_row], positions[second_row],
                                bool(rng.integers(2)))
            if fitted is None:
                continue
            matrix, translation = fitted
            singular = np.linalg.svd(matrix, compute_uv=False)
            if singular[-1] < config.min_scale or singular[0] > config.max_scale:
                continue
            score, matches = _score_pose(
                matrix, translation, nodes, positions, weights, patch_index,
                tree, config, required_support=2 if two_node_fallback else None,
            )
            if not np.isfinite(score):
                continue
            if two_node_fallback:
                # A two-star figure cannot corroborate the two-point pose.
                # Keep it available only as an explicitly weak fallback.
                score = -0.5 + 0.05 * sum(match[2] for match in matches)
            if len(top) < config.keep_proposals or score > top[-1][0]:
                top.append((score, matrix, translation, matches))
                top.sort(key=lambda item: item[0], reverse=True)
                del top[config.keep_proposals:]
        if not top:
            records.append({"pattern": name, "score": -np.inf, "support": 0,
                            "status": "no_consensus"})
            continue
        if two_node_fallback:
            refined = top
        else:
            triangle_hypotheses = _triangle_hypotheses(
                top, nodes, positions, weights, patch_index, tree, config,
            )
            refined = []
            for _, matrix, translation, _ in top + triangle_hypotheses:
                exact_score, exact_matches = _score_pose(
                    matrix, translation, nodes, positions, weights,
                    patch_index, tree, config, exact_assignment=True,
                )
                if np.isfinite(exact_score):
                    refined.append(_refine(
                        (exact_score, matrix, translation, exact_matches),
                        nodes, positions, weights, patch_index, tree, config,
                    ))
        best = max(refined, key=lambda item: (item[0], len(item[3])))
        winners[name] = best
        score, matrix, translation, matches = best
        null_mean = 0.0 if two_node_fallback else _shifted_null_mean(
            matrix, translation, nodes, positions, weights, patch_index,
            tree, config,
        )
        winner_nulls[name] = null_mean
        adjusted_score = score - config.null_shift_weight * null_mean
        records.append({"pattern": name, "score": adjusted_score,
                        "raw_score": score, "shifted_null_mean": null_mean,
                        "support": len(matches),
                        "weighted_support": sum(m[2] for m in matches),
                        "median_residual_px": float(np.median([m[3] for m in matches])),
                        "reflection": bool(np.linalg.det(matrix) < 0),
                        "status": ("two_node_weak" if two_node_fallback
                                   else "fitted")})

    rankings = pd.DataFrame(records).sort_values(
        ["score", "support", "pattern"], ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    best_name = str(rankings.iloc[0]["pattern"])
    if best_name not in winners:
        return JointResult(rankings, _empty_selections(b), None)
    score, matrix, translation, matches = winners[best_name]
    match_lookup = {patch_ids[patch_index[row]]: (row, node, strength, residual)
                    for row, node, strength, residual in matches}
    baseline_lookup = b.set_index("patch_id")
    selection_records = []
    for patch_id in b["patch_id"]:
        baseline_row = baseline_lookup.loc[patch_id]
        match = match_lookup.get(patch_id)
        row = c.iloc[match[0]] if match else None
        selection_records.append({
            "patch_id": patch_id,
            "x": row["x"] if match else baseline_row["x"],
            "y": row["y"] if match else baseline_row["y"],
            "geom_x": row["geom_x"] if match else np.nan,
            "geom_y": row["geom_y"] if match else np.nan,
            "matched": match is not None,
            "node_index": match[1] if match else -1,
            "match_weight": match[2] if match else 0.0,
            "residual_px": match[3] if match else np.nan,
            "baseline_x": baseline_row["x"],
            "baseline_y": baseline_row["y"],
            "baseline_p": baseline_row.get("p", baseline_row.get("present", np.nan)),
            "baseline_m": baseline_row.get("m", baseline_row.get("member", np.nan)),
        })
    selections = pd.DataFrame(selection_records)
    pose = {"pattern": best_name,
            "score": score - config.null_shift_weight * winner_nulls[best_name],
            "raw_score": score,
            "shifted_null_mean": winner_nulls[best_name],
            "tentative": len(pattern_nodes[best_name]) == 2,
            "matrix": matrix,
            "translation": translation, "support": len(matches),
            "reflection": bool(np.linalg.det(matrix) < 0)}
    return JointResult(rankings, selections, pose)


def _empty_selections(baseline: pd.DataFrame) -> pd.DataFrame:
    result = baseline[["patch_id", "x", "y"]].copy()
    result["geom_x"] = np.nan
    result["geom_y"] = np.nan
    result["matched"] = False
    result["node_index"] = -1
    result["match_weight"] = 0.0
    result["residual_px"] = np.nan
    result["baseline_x"] = result["x"]
    result["baseline_y"] = result["y"]
    result["baseline_p"] = baseline.get("p", baseline.get("present", np.nan))
    result["baseline_m"] = baseline.get("m", baseline.get("member", np.nan))
    return result
