"""Training-label-only analyses; never used for test-region prediction."""
from pathlib import Path
import numpy as np
from .data import read_rows, parse_instances, image_shape, image_path, split_id
from .matching import fit_similarity


def geometry_diagnostic(data_root):
    root = Path(data_root)
    rows = read_rows(root / "training/train_ground_truth.csv")
    reports = []
    for sid, row in rows.items():
        centers = {}
        for modality in ("invivo", "exvivo"):
            shape = image_shape(image_path(root, "training", sid, modality))
            centers[modality] = {key: mask.coordinates().mean(0) for key, mask in parse_instances(row, modality, shape).items() if mask.area}
        pairs = row["match_pairs"]
        if len(pairs) < 3:
            continue
        source = np.array([centers["invivo"][i] for i, _ in pairs])
        target = np.array([centers["exvivo"][e] for _, e in pairs])
        reflected = fit_similarity(source, target, True)
        proper = fit_similarity(source, target, False)
        design = np.column_stack([source, np.ones(len(source))])
        affine = np.linalg.lstsq(design, target, rcond=None)[0]
        reports.append({"sample_id": sid, "pairs": len(pairs), "similarity_scale": reflected[2],
                        "reflected": bool(np.linalg.det(reflected[0]) < 0),
                        "proper_similarity_median_error": float(np.median(np.linalg.norm(source @ proper[0] + proper[1] - target, axis=1))),
                        "similarity_median_error": float(np.median(np.linalg.norm(source @ reflected[0] + reflected[1] - target, axis=1))),
                        "affine_median_error": float(np.median(np.linalg.norm(design @ affine - target, axis=1))),
                        "affine_singular_values": np.linalg.svd(affine[:2], compute_uv=False).tolist()})
    summary = {}
    for subject in sorted({split_id(row["sample_id"])[0] for row in reports}):
        subset = [row for row in reports if split_id(row["sample_id"])[0] == subject]
        summary[subject] = {"regions": len(subset),
                            "median_similarity_error": float(np.median([r["similarity_median_error"] for r in subset])),
                            "median_affine_error": float(np.median([r["affine_median_error"] for r in subset])),
                            "median_scale": float(np.median([r["similarity_scale"] for r in subset]))}
    return {"purpose": "Fit known training correspondences to diagnose geometry. This is not inference or held-out performance. Residuals are measured on the same pairs used for fitting; affine has more parameters.",
            "by_subject": summary, "regions": reports}
