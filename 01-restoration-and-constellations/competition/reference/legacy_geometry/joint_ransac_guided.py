"""Deterministic, candidate-aware constellation seeding.

This is an experimental alternative to the random two-point proposals in
``joint_ransac.py``. It never reads labels. For selected query pairs it tests
all retained location pairs against all sufficiently separated template node
pairs. A fast nearest-neighbor score screens poses; exact distinct-query
assignment and the original refinement/scoring functions make the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from joint_ransac import (
    JointConfig, JointResult, _check_points, _empty_selections, _prepare,
    _refine, _score_pose, _shifted_null_mean,
)


@dataclass(frozen=True)
class GuidedConfig:
    joint: JointConfig = JointConfig()
    patch_pairs: int = 128
    hypotheses_per_node_pair: int = 3
    exact_hypotheses: int = 100
    chunk_size: int = 1024


def _observed_pairs(patch_index: np.ndarray,
                    patch_count: int, config: GuidedConfig,
                    preferred_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select query pairs, then include every retained candidate combination."""
    first, second = np.triu_indices(patch_count, 1)
    priority = preferred_patches[first] * preferred_patches[second]
    # One common observed-pair sample is used for every class so the ranking
    # cannot improve for a template merely because it drew luckier query pairs.
    rng = np.random.default_rng(config.joint.seed)
    # The random keys select pairs without replacement and independently of
    # input row order. A small preference for baseline-predicted members is
    # permitted; these are model predictions rather than ground-truth labels.
    keys = -np.log(np.clip(rng.random(len(first)), 1e-12, 1)) / priority
    chosen = np.argsort(keys, kind='stable')[:config.patch_pairs]
    row_groups = [np.flatnonzero(patch_index == i) for i in range(patch_count)]
    a = np.concatenate([np.repeat(row_groups[first[k]], len(row_groups[second[k]]))
                        for k in chosen]).astype(np.int32)
    b = np.concatenate([np.tile(row_groups[second[k]], len(row_groups[first[k]]))
                        for k in chosen]).astype(np.int32)
    return a, b


def _cheap_scores(mapped: np.ndarray, tree: cKDTree, weights: np.ndarray,
                  patch_index: np.ndarray, radius: float, sigma: float,
                  chunk_size: int, required_support: int) -> np.ndarray:
    """Screen hypotheses by independently located nodes and unique queries.

    The exact one-to-one assignment is intentionally deferred until after
    screening. The cheap score is only an upper-level ordering, never the
    final pattern score.
    """
    output = np.full(len(mapped), -np.inf, dtype=float)
    for start in range(0, len(mapped), chunk_size):
        stop = min(start + chunk_size, len(mapped))
        dist, nearest = tree.query(mapped[start:stop].reshape(-1, 2), workers=1)
        dist = dist.reshape(stop - start, mapped.shape[1])
        nearest = nearest.reshape(stop - start, mapped.shape[1])
        valid = dist <= radius
        # Keeping only the best node for a repeated query is a cheap proxy for
        # the later exact bipartite assignment.
        strength = weights[nearest] * np.exp(-0.5 * (dist / sigma) ** 2)
        strength[~valid] = 0
        patch_ids = patch_index[nearest]
        order = np.argsort(-strength, axis=1, kind='stable')
        patches_sorted = np.take_along_axis(patch_ids, order, axis=1)
        strength_sorted = np.take_along_axis(strength, order, axis=1)
        for column in range(1, strength_sorted.shape[1]):
            repeated = (patches_sorted[:, column, None] ==
                        patches_sorted[:, :column]).any(axis=1)
            strength_sorted[repeated, column] = 0
        support = (strength_sorted > 0).sum(axis=1)
        # The two seed correspondences are tautological and receive no credit.
        value = strength_sorted[:, 2:].sum(axis=1) + 0.08 * (support - 2)
        output[start:stop] = np.where(support >= required_support, value, -np.inf)
    return output


def _guided_hypotheses(nodes: np.ndarray, positions: np.ndarray,
                       weights: np.ndarray, patch_index: np.ndarray,
                       tree: cKDTree, observed: tuple[np.ndarray, np.ndarray],
                       config: GuidedConfig):
    cfg = config.joint
    source = nodes[:, 0] + 1j * nodes[:, 1]
    target = positions[:, 0] + 1j * positions[:, 1]
    distances = abs(source[:, None] - source[None, :])
    i_nodes, j_nodes = np.where(np.triu(
        distances >= cfg.min_node_pair_fraction * distances.max(), 1))
    first, second = observed
    proposed = []
    for i, j in zip(i_nodes, j_nodes):
        for reflected in (False, True):
            pattern = source.conj() if reflected else source
            delta = pattern[j] - pattern[i]
            for a_rows, b_rows in ((first, second), (second, first)):
                for start in range(0, len(a_rows), config.chunk_size):
                    a = a_rows[start:start + config.chunk_size]
                    b = b_rows[start:start + config.chunk_size]
                    alpha = (target[b] - target[a]) / delta
                    valid = ((abs(alpha) >= cfg.min_scale) &
                             (abs(alpha) <= cfg.max_scale))
                    if not valid.any():
                        continue
                    a, b, alpha = a[valid], b[valid], alpha[valid]
                    beta = target[a] - alpha * pattern[i]
                    mapped = alpha[:, None] * pattern[None, :] + beta[:, None]
                    mapped_xy = np.stack((mapped.real, mapped.imag), axis=-1)
                    cheap = _cheap_scores(mapped_xy, tree, weights, patch_index,
                                          cfg.inlier_radius, cfg.gaussian_sigma,
                                          config.chunk_size,
                                          2 if len(nodes) == 2 else 3)
                    if not np.isfinite(cheap).any():
                        continue
                    count = min(config.hypotheses_per_node_pair, len(cheap))
                    top = np.argpartition(cheap, -count)[-count:]
                    for k in top:
                        if np.isfinite(cheap[k]):
                            proposed.append((float(cheap[k]), complex(alpha[k]),
                                             complex(beta[k]), reflected))
    proposed.sort(key=lambda value: value[0], reverse=True)
    # Similar seeds can crowd the shortlist; bin the transformed template
    # centroid and linear map before exact assignment.
    seen = set()
    unique = []
    centroid = source.mean()
    for quick, alpha, beta, reflected in proposed:
        mapped_center = alpha * (centroid.conjugate() if reflected else centroid) + beta
        key = (round(mapped_center.real / cfg.inlier_radius),
               round(mapped_center.imag / cfg.inlier_radius),
               round(alpha.real / max(abs(alpha) * .08, .01)),
               round(alpha.imag / max(abs(alpha) * .08, .01)), reflected)
        if key in seen:
            continue
        seen.add(key)
        unique.append((quick, alpha, beta, reflected))
        if len(unique) >= config.exact_hypotheses:
            break
    exact = []
    for _, alpha, beta, reflected in unique:
        # z -> alpha * z (+ conjugation under reflection) + beta.
        matrix = np.array([[alpha.real, alpha.imag if reflected else -alpha.imag],
                           [alpha.imag, -alpha.real if reflected else alpha.real]])
        translation = np.array([beta.real, beta.imag])
        score, matches = _score_pose(
            matrix, translation, nodes, positions, weights, patch_index,
            tree, cfg, exact_assignment=True,
            required_support=2 if len(nodes) == 2 else None)
        if np.isfinite(score):
            exact.append((score, matrix, translation, matches))
    exact.sort(key=lambda value: value[0], reverse=True)
    return exact[:cfg.keep_proposals]


def _prepare_guided(candidates: pd.DataFrame,
                    baseline: pd.DataFrame | None,
                    config: GuidedConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain each image-derived saved pose, including two protected sources.

    The shared base preparation reserves one old pose. A validation bank can
    contain both ``saved_dense_phase`` and ``v6_submission``; omitting either
    can eliminate the correct position before geometric fitting.
    """
    c, b = _prepare(candidates, baseline, config.joint)
    if 'source' not in candidates:
        return c, b
    score_col = 'appearance_score' if 'appearance_score' in candidates else 'score'
    extras = []
    for patch_id, group in candidates[candidates['source'].ne('retrieved')].groupby('patch_id'):
        retained = set(c.loc[c.patch_id.eq(patch_id), 'source'])
        for source, source_group in group.groupby('source', sort=False):
            if source in retained:
                continue
            row = source_group.sort_values(score_col, ascending=False).iloc[0].copy()
            row['appearance_score'] = float(row[score_col])
            row['geom_x'] = float(row.get('geom_x', row['x']))
            row['geom_y'] = float(row.get('geom_y', row['y']))
            best = float(c.loc[c.patch_id.eq(patch_id), 'appearance_score'].max())
            relative = np.exp(np.clip((row['appearance_score'] - best) /
                                      config.joint.candidate_temperature, -12., 0.))
            weight = .25 + .75 * relative
            if 'presence_prob' in row and pd.notna(row['presence_prob']):
                weight *= .5 + .5 * np.clip(float(row['presence_prob']), 0., 1.)
            row['_weight'] = weight
            extras.append(row)
    if extras:
        c = pd.concat((c, pd.DataFrame(extras)), ignore_index=True)
        c['_row_id'] = np.arange(len(c), dtype=np.int32)
    return c, b


def fit_joint_constellations_guided(
    candidates: pd.DataFrame,
    pattern_nodes: Mapping[str, np.ndarray],
    baseline: pd.DataFrame | None = None,
    config: GuidedConfig = GuidedConfig(),
) -> JointResult:
    """Fit classes with geometric pair enumeration and exact query assignment."""
    if not pattern_nodes:
        raise ValueError('No patterns supplied')
    if config.patch_pairs < 1 or config.hypotheses_per_node_pair < 1:
        raise ValueError('Positive proposal limits are required')
    c, b = _prepare_guided(candidates, baseline, config)
    patch_ids = c['patch_id'].drop_duplicates().tolist()
    patch_lookup = {patch_id: i for i, patch_id in enumerate(patch_ids)}
    patch_index = c['patch_id'].map(patch_lookup).to_numpy(np.int32)
    positions = c[['geom_x', 'geom_y']].to_numpy(float)
    weights = c['_weight'].to_numpy(float)
    tree = cKDTree(positions)
    # Use the same image-only pair distribution in train and validation.
    # The available train baseline has no membership column, whereas the v6
    # validation submission does; using it here would change the seeding
    # process across splits without a way to measure that change fairly.
    preferred = np.ones(len(patch_ids), dtype=float)

    observed = _observed_pairs(patch_index, len(patch_ids), config, preferred)
    rankings = []
    winners = {}
    for name, raw_nodes in sorted(pattern_nodes.items()):
        nodes = _check_points(raw_nodes)
        proposals = _guided_hypotheses(nodes, positions, weights, patch_index,
                                       tree, observed, config)
        if not proposals:
            rankings.append({'pattern': name, 'score': -np.inf,
                             'support': 0, 'status': 'no_consensus'})
            continue
        refined = (proposals if len(nodes) == 2 else
                   [_refine(item, nodes, positions, weights, patch_index,
                            tree, config.joint) for item in proposals])
        score, matrix, translation, matches = max(
            refined, key=lambda item: (item[0], len(item[3])))
        null_mean = (0.0 if len(nodes) == 2 else
                     _shifted_null_mean(matrix, translation, nodes, positions,
                                        weights, patch_index, tree, config.joint))
        if len(nodes) == 2:
            # Two exact seed correspondences have no corroborating point.
            score = -.5 + .05 * sum(match[2] for match in matches)
        winners[name] = (score, matrix, translation, matches, null_mean)
        adjusted = (score - config.joint.null_shift_weight * null_mean)
        # A pattern with many nodes has more opportunities to fit unrelated
        # background stars. Require that a substantial fraction of its nodes
        # are explained, in addition to obtaining high distinct-query support.
        coverage = len(matches) / len(nodes)
        adjusted *= coverage
        rankings.append({'pattern': name, 'score': adjusted, 'raw_score': score,
                         'shifted_null_mean': null_mean, 'support': len(matches),
                         'node_count': len(nodes), 'coverage_fraction': coverage,
                         'weighted_support': sum(m[2] for m in matches),
                         'median_residual_px': float(np.median([m[3] for m in matches])),
                         'reflection': bool(np.linalg.det(matrix) < 0),
                         'status': 'fitted' if len(nodes) > 2 else 'two_node_weak'})
    ranking_frame = pd.DataFrame(rankings).sort_values(
        ['score', 'support', 'pattern'], ascending=[False, False, True],
        kind='stable').reset_index(drop=True)
    best_name = str(ranking_frame.iloc[0]['pattern'])
    if best_name not in winners:
        return JointResult(ranking_frame, _empty_selections(b), None)
    score, matrix, translation, matches, null_mean = winners[best_name]
    lookup = {patch_ids[patch_index[row]]: (row, node, strength, residual)
              for row, node, strength, residual in matches}
    baseline_lookup = b.set_index('patch_id')
    selection_records = []
    for patch_id in b['patch_id']:
        base = baseline_lookup.loc[patch_id]
        match = lookup.get(patch_id)
        row = c.iloc[match[0]] if match else None
        selection_records.append({
            'patch_id': patch_id,
            'x': row['x'] if match else base['x'],
            'y': row['y'] if match else base['y'],
            'geom_x': row['geom_x'] if match else np.nan,
            'geom_y': row['geom_y'] if match else np.nan,
            'matched': match is not None,
            'node_index': match[1] if match else -1,
            'match_weight': match[2] if match else 0.,
            'residual_px': match[3] if match else np.nan,
            'baseline_x': base['x'], 'baseline_y': base['y'],
            'baseline_p': base.get('p', base.get('present', np.nan)),
            'baseline_m': base.get('m', base.get('member', np.nan)),
        })
    selections = pd.DataFrame(selection_records)
    pose = {'pattern': best_name,
            'score': float(ranking_frame.iloc[0]['score']),
            'raw_score': score, 'shifted_null_mean': null_mean,
            'tentative': len(pattern_nodes[best_name]) == 2,
            'matrix': matrix, 'translation': translation,
            'support': len(matches),
            'reflection': bool(np.linalg.det(matrix) < 0)}
    return JointResult(ranking_frame, selections, pose)
