"""Triangle-hash similarity registration and optional one-to-one assignment.

This is a geometry-only baseline, not a learned correspondence model. All
distances below are in original ex-vivo pixels. Partial overlap is allowed.
"""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.optimize import linear_sum_assignment


def centroids(masks):
    keys, points = [], []
    for key, mask in masks.items():
        if mask.area:
            keys.append(key)
            points.append(mask.coordinates().mean(axis=0))
    return keys, np.asarray(points, dtype=float).reshape(-1, 2)


def triangles(points, neighbors, limit, rng):
    k = min(neighbors + 1, len(points))
    indices = cKDTree(points).query(points, k=k)[1]
    unique = set()
    for i, row in enumerate(indices):
        for j, m in combinations([int(v) for v in row if v != i], 2):
            unique.add(tuple(sorted((i, j, m))))
    candidates = sorted(unique)
    if len(candidates) > limit:
        candidates = [candidates[i] for i in sorted(rng.choice(len(candidates), limit, replace=False))]
    descriptors, vertices = [], []
    for ids in candidates:
        p = points[list(ids)]
        opposite = np.array([np.linalg.norm(p[1]-p[2]), np.linalg.norm(p[0]-p[2]), np.linalg.norm(p[0]-p[1])])
        order = np.argsort(opposite)
        sides = opposite[order]
        a, b = p[1]-p[0], p[2]-p[0]
        twice_area = abs(a[0]*b[1]-a[1]*b[0])
        if sides[-1] <= 0 or twice_area / sides[-1]**2 < 0.08:
            continue
        descriptors.append(sides[:2] / sides[-1])
        vertices.append(np.asarray(ids)[order])
    return np.asarray(descriptors).reshape(-1, 2), vertices


def fit_similarity(source, target, allow_reflection=False):
    x, y = source-source.mean(axis=0), target-target.mean(axis=0)
    denominator = np.sum(x*x)
    if denominator <= 1e-12:
        return None
    u, singular, vt = np.linalg.svd(x.T @ y)
    signs = np.ones(2)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    scale = float(np.sum(singular*signs) / denominator)
    matrix = scale * (u @ np.diag(signs) @ vt)
    offset = target.mean(axis=0) - source.mean(axis=0) @ matrix
    return matrix, offset, scale


def mutual_neighbors(source, target, radius):
    distances, nearest = cKDTree(target).query(source)
    back = cKDTree(source).query(target)[1]
    src = np.arange(len(source))
    keep = (distances < radius) & (back[nearest] == src)
    return src[keep], nearest[keep], distances[keep]


def match_cells(iv_masks, ex_masks, config, seed=2026):
    iv_ids, source = centroids(iv_masks)
    ex_ids, target = centroids(ex_masks)
    empty = {"status": "insufficient_registration_support", "inliers": 0}
    if min(len(source), len(target)) < max(3, config["min_inliers"]):
        return [], empty
    rng = np.random.default_rng(seed)
    desc_a, triangles_a = triangles(source, config["neighbors"], config["max_triangles"], rng)
    desc_b, triangles_b = triangles(target, config["neighbors"], config["max_triangles"], rng)
    if not len(desc_a) or not len(desc_b):
        return [], empty
    distances, neighbors = cKDTree(desc_b).query(desc_a, k=min(3, len(desc_b)))
    distances, neighbors = distances.reshape(len(desc_a), -1), neighbors.reshape(len(desc_a), -1)
    candidates = [(float(distances[i, j]), i, int(neighbors[i, j]))
                  for i, j in zip(*np.where(distances < config["triangle_tolerance"]))]
    # Mix the closest descriptors and seeded random hypotheses to avoid a
    # budget dominated by nearly equilateral, ambiguous triangles.
    candidates.sort()
    budget = config["max_hypotheses"]
    if len(candidates) > budget:
        first = candidates[:budget//2]
        rest = candidates[budget//2:]
        candidates = first + [rest[i] for i in rng.choice(len(rest), budget-len(first), replace=False)]
    best, best_key = None, (-1, -float("inf"))
    for _, a, b in candidates:
        transform = fit_similarity(source[triangles_a[a]], target[triangles_b[b]], config["allow_reflection"])
        if transform is None or not config["min_scale"] <= transform[2] <= config["max_scale"]:
            continue
        matrix, offset, _ = transform
        src, dst, errors = mutual_neighbors(source @ matrix + offset, target, config["registration_radius"])
        key = (len(src), -float(np.median(errors)) if len(errors) else -float("inf"))
        if key > best_key:
            best, best_key = transform, key
    minimum = max(config["min_inliers"], int(np.ceil(config["min_inlier_fraction"] * min(len(source), len(target)))))
    if best is None or best_key[0] < minimum:
        return [], empty
    for _ in range(5):
        matrix, offset, _ = best
        src, dst, errors = mutual_neighbors(source @ matrix + offset, target, config["registration_radius"])
        if len(src) < minimum:
            break
        refined = fit_similarity(source[src], target[dst], config["allow_reflection"])
        if refined is None or not config["min_scale"] <= refined[2] <= config["max_scale"]:
            break
        ri, _, re = mutual_neighbors(source @ refined[0] + refined[1], target, config["registration_radius"])
        if (len(ri), -float(np.median(re)) if len(re) else -float("inf")) < (len(src), -float(np.median(errors))):
            break
        best = refined
    matrix, offset, scale = best
    transformed = source @ matrix + offset
    costs = distance.cdist(transformed, target)
    radius = config["match_radius"]
    # Each source has a private dummy column, so no real match is forced.
    augmented = np.full((len(source), len(target)+len(source)), radius)
    augmented[:, :len(target)] = costs
    rows, cols = linear_sum_assignment(augmented)
    pairs = []
    for i, j in zip(rows, cols):
        if j >= len(target) or costs[i, j] >= radius:
            continue
        other_row = np.delete(costs[i], j)
        other_col = np.delete(costs[:, j], i)
        alternative = min(float(other_row.min()) if len(other_row) else np.inf,
                          float(other_col.min()) if len(other_col) else np.inf)
        if costs[i, j] > config["ambiguity_ratio"] * alternative:
            continue
        pairs.append([iv_ids[i], ex_ids[j]])
    inliers, _, errors = mutual_neighbors(transformed, target, config["registration_radius"])
    return pairs, {"status": "registered", "inliers": len(inliers), "hypotheses": len(candidates),
                   "median_residual": float(np.median(errors)), "scale": scale,
                   "matrix": matrix.tolist(), "offset": offset.tolist(), "accepted_pairs": len(pairs)}
