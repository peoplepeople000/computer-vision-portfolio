"""Local implementation of the published metric; not organizer-supplied code."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def segmentation_score(predicted, truth, threshold=0.75):
    pids, gids = list(predicted), list(truth)
    ious = np.zeros((len(pids), len(gids)), dtype=float)
    for i, pid in enumerate(pids):
        for j, gid in enumerate(gids):
            ious[i, j] = predicted[pid].iou(truth[gid])
    eligible = ious > threshold  # Strictly greater, never >=.
    # Maximize valid match count first, then total IoU for deterministic ambiguity resolution.
    weights = eligible * (min(ious.shape, default=0) + 1 + ious)
    pi, gi = linear_sum_assignment(weights, maximize=True)
    good = eligible[pi, gi]
    pi, gi = pi[good], gi[good]
    tp = len(pi)
    fp, fn = len(pids) - tp, len(gids) - tp
    total_iou = float(ious[pi, gi].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": total_iou / denominator if denominator else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "iou_sum": total_iou,
            "pred_to_gt": {pids[i]: gids[j] for i, j in zip(pi, gi)}}


def pair_score(pairs, verified, iv_map, ex_map):
    verified = {tuple(pair) for pair in verified}
    correct = sum((iv_map.get(iv), ex_map.get(ex)) in verified for iv, ex in pairs)
    return {"correct": correct, "predicted": len(pairs), "verified": len(verified)}


def aggregate(regions):
    if not regions:
        raise ValueError("Cannot aggregate zero regions")
    iv = float(np.mean([r["invivo"]["pq"] for r in regions]))
    ex = float(np.mean([r["exvivo"]["pq"] for r in regions]))
    correct = sum(r["matching"]["correct"] for r in regions)
    predicted = sum(r["matching"]["predicted"] for r in regions)
    verified = sum(r["matching"]["verified"] for r in regions)
    precision = correct / predicted if predicted else 0.0
    recall = correct / verified if verified else 0.0
    f1 = 2 * correct / (predicted + verified) if predicted + verified else 0.0
    return {"regions": len(regions), "pq_invivo": iv, "pq_exvivo": ex,
            "matching_precision": precision, "matching_recall": recall, "matching_f1": f1,
            "correct_pairs": correct, "predicted_pairs": predicted, "verified_pairs": verified,
            "score": 0.25 * iv + 0.25 * ex + 0.5 * f1}
