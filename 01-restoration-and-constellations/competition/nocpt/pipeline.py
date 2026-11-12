"""Combine patch evidence and constellation geometry into scene predictions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .classifier import CandidateClassifier, feature_vector
from .data import format_cell
from .geometry import PatternResult, rank_patterns
from .matching import Candidate
from .subpixel import refine_center


def _star_profile(image: np.ndarray, candidate: Candidate) -> np.ndarray | None:
    x, y = round(candidate.x), round(candidate.y)
    if not (7 <= x < image.shape[1]-7 and 7 <= y < image.shape[0]-7):
        return None
    crop = image[y-7:y+8, x-7:x+8].astype(np.float32)
    crop -= cv2.GaussianBlur(crop, (0, 0), 3)
    vector = crop.ravel()
    vector -= vector.mean()
    vector /= max(1e-6, float(np.linalg.norm(vector)))
    return vector


def _select_for_pattern(pattern: PatternResult, image: np.ndarray,
                        options: dict[int, tuple[list[Candidate], np.ndarray]],
                        selected: dict[int, Candidate]) -> tuple[dict[int, Candidate], list[dict]]:
    tree = cKDTree(pattern.projected)
    reference = []
    for candidate in selected.values():
        if tree.query([candidate.x, candidate.y])[0] <= 20:
            profile = _star_profile(image, candidate)
            if profile is not None:
                reference.append(profile)
    if len(reference) < 3:
        return selected.copy(), []
    reference_matrix = np.stack(reference)
    result = selected.copy()
    changes = []
    occupancy: dict[int, set[int]] = {}
    for index, candidate in selected.items():
        distance, node = tree.query([candidate.x, candidate.y])
        if distance <= 20:
            occupancy.setdefault(int(node), set()).add(index)
    for number, (candidates, probabilities) in options.items():
        old = selected.get(number)
        old_distance, old_node = (tree.query([old.x, old.y]) if old is not None
                                  else (float("inf"), -1))
        duplicate = old is not None and len(occupancy.get(int(old_node), set())) > 1
        if old is not None and old_distance <= 20 and not duplicate:
            continue
        choices = []
        for rank, candidate in enumerate(candidates):
            distance, node = tree.query([candidate.x, candidate.y])
            distance = float(distance)
            node = int(node)
            if distance > 17 or node == old_node or candidate.fine_score < 0.90:
                continue
            node_occupied = bool(occupancy.get(node, set()) - {number})
            profile = _star_profile(image, candidate)
            if profile is None:
                continue
            coherence = float(np.median(reference_matrix @ profile))
            probability = float(probabilities[rank])
            best_probability = float(probabilities.max())
            regular = (coherence >= 0.70 and candidate.confidence >= 1.10 and
                       probability >= 0.40 and best_probability-probability <= 0.20)
            duplicate_rescue = (duplicate and coherence >= 0.80 and
                                not node_occupied and candidate.confidence >= 1.05 and
                                candidate.fine_score >= 0.94)
            saturated_rescue = (candidate.confidence == 0 and candidate.fine_score >= 0.95 and
                                coherence >= 0.80 and old is not None and not node_occupied)
            if regular or duplicate_rescue or saturated_rescue:
                choices.append((rank, candidate, distance, probability, coherence,
                                node, node_occupied))
        if not choices:
            continue
        rank, candidate, distance, probability, coherence, node, _ = max(
            choices, key=lambda item: 0.8*item[3] + 0.2*item[4]
            - 0.005*item[2] - 0.18*item[6])
        if old is not None and old_distance <= 20:
            occupancy[int(old_node)].discard(number)
        occupancy.setdefault(node, set()).add(number)
        result[number] = candidate
        changes.append({"patch": number, "candidate_rank": rank,
                        "distance_to_node": distance, "probability": probability,
                        "star_shape_coherence": coherence,
                        "previously_present": old is not None})
    return result, changes


def _render_overlay(image: np.ndarray, selected: dict[int, Candidate],
                    pattern: PatternResult | None, path: Path) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if pattern is not None:
        for x, y in pattern.projected:
            if 0 <= x < 3000 and 0 <= y < 3000:
                cv2.circle(canvas, (round(x), round(y)), 12, (0, 210, 0), 2)
    for number, candidate in selected.items():
        centre = (round(candidate.x), round(candidate.y))
        cv2.circle(canvas, centre, 9, (0, 0, 255), 2)
        cv2.putText(canvas, f"{number:02}", (centre[0]+11, centre[1]-11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    canvas = cv2.resize(canvas, (1500, 1500), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"Predicted: {pattern.name if pattern else 'unknown'}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def predict_scene(root: Path, split: str, row: dict[str, str],
                  candidates: dict[int, list[Candidate]],
                  patterns: dict[str, np.ndarray], model: CandidateClassifier,
                  threshold: float, diagnostics: Path,
                  target_present_fraction: float = 0.61,
                  prevalence_strength: float = 0.0,
                  min_present_fraction: float = 0.0) -> dict[str, str]:
    scene = row["Id"]
    folder = root / split / scene
    image = cv2.imread(str(folder / f"{scene}_image.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(folder / f"{scene}_image.png")
    options: dict[int, tuple[list[Candidate], np.ndarray]] = {}
    selected: dict[int, Candidate] = {}
    for number in range(1, int(row["n_patches"]) + 1):
        patch = cv2.imread(str(folder / "patches" / f"patch_{number:02}.png"), 0)
        choices = candidates[number]
        if not choices:
            options[number] = ([], np.asarray([]))
            continue
        features = np.stack([model.extract(image, patch, item, rank)
                             for rank, item in enumerate(choices)])
        probabilities = model.predict(features)
        options[number] = (choices, probabilities)
    if prevalence_strength:
        maxima = [float(probabilities.max()) for _, probabilities in options.values()
                  if len(probabilities)]
        prevalence_cutoff = max(0.40, float(np.quantile(maxima,
                                                      1-target_present_fraction)))
        threshold = (1-prevalence_strength)*threshold + prevalence_strength*prevalence_cutoff
    for number, (choices, probabilities) in options.items():
        if not len(probabilities):
            continue
        winner = int(np.argmax(probabilities))
        if probabilities[winner] >= threshold:
            selected[number] = choices[winner]
    ranked = rank_patterns(patterns, {number: [item] for number, item in selected.items()}, image)
    naming_fallback = False
    if not ranked:
        # A heavily degraded scene can have too few accepted patches to name.
        # Use the strongest additional image-verified matches for naming only.
        tentative = sorted(
            ((float(probabilities.max()), number,
              choices[int(np.argmax(probabilities))])
             for number, (choices, probabilities) in options.items() if len(probabilities)),
            reverse=True, key=lambda entry: entry[0])
        seeds = {number: [candidate] for probability, number, candidate in tentative[:12]
                 if probability >= 0.10}
        ranked = rank_patterns(patterns, seeds, image, min_evidence=0.30)
        naming_fallback = bool(ranked)
    changes: list[dict] = []
    hypothetical_changes: list[dict] = []
    # Evaluate limited global hypotheses before committing to a constellation.
    if ranked:
        alternatives = []
        for pattern in ranked[:3]:
            corrected, corrections = _select_for_pattern(pattern, image, options, selected)
            adjusted_score = pattern.score + min(4.0, 1.4*len(corrections))
            alternatives.append((adjusted_score, pattern, corrected, corrections))
        alternatives.sort(key=lambda item: item[0], reverse=True)
        _, leading, proposed, hypothetical_changes = alternatives[0]
        tree = cKDTree(leading.projected)
        original_occupancy: dict[int, int] = {}
        for candidate in selected.values():
            distance, node = tree.query([candidate.x, candidate.y])
            if distance <= 20:
                original_occupancy[int(node)] = original_occupancy.get(int(node), 0) + 1
        for change in hypothetical_changes:
            number = change["patch"]
            old = selected.get(number)
            if old is None:
                continue
            candidate = proposed[number]
            new_distance, new_node = tree.query([candidate.x, candidate.y])
            old_distance, old_node = tree.query([old.x, old.y])
            collision = any(
                other_number != number and np.hypot(candidate.x-other.x,
                                                    candidate.y-other.y) < 12
                for other_number, other in selected.items())
            probabilities = options[number][1]
            gap = float(probabilities.max())-change["probability"]
            close_evidence = gap <= 0.12 and not collision
            duplicate_resolution = (
                old_distance <= 20 and original_occupancy.get(int(old_node), 0) > 1
                and new_distance <= 13 and original_occupancy.get(int(new_node), 0) == 0
                and candidate.confidence >= 1.05 and candidate.fine_score >= 0.94
                and change["star_shape_coherence"] >= 0.80 and not collision)
            saturated_rescue = (
                candidate.confidence == 0 and candidate.fine_score >= 0.95
                and change["star_shape_coherence"] >= 0.80 and not collision)
            if close_evidence or duplicate_resolution or saturated_rescue:
                selected[number] = candidate
                changes.append(change)
        if changes:
            ranked = rank_patterns(patterns, {number: [item] for number, item in selected.items()}, image)
        chosen_pattern = next((item for item in ranked if item.name == leading.name), leading)
    else:
        chosen_pattern = None
    initial_present_fraction = len(selected) / max(1, int(row["n_patches"]))
    presence_backfill: list[int] = []
    if min_present_fraction:
        minimum = int(np.ceil(min_present_fraction * int(row["n_patches"])))
        ordered = sorted(
            ((float(probabilities.max()), number,
              choices[int(np.argmax(probabilities))])
             for number, (choices, probabilities) in options.items()
             if len(probabilities) and number not in selected),
            reverse=True, key=lambda entry: entry[0])
        for probability, number, candidate in ordered:
            if len(selected) >= minimum or probability < 0.05:
                break
            if any(np.hypot(candidate.x-other.x, candidate.y-other.y) < 12
                   for other in selected.values()):
                continue
            selected[number] = candidate
            presence_backfill.append(number)
    reranking_after_backfill = False
    initial_margin = (ranked[0].score-ranked[1].score if len(ranked) >= 2
                      else float("inf") if ranked else 0.0)
    if (presence_backfill and initial_present_fraction < 0.35
            and (chosen_pattern is None or initial_margin < 5.0)):
        relaxed = rank_patterns(patterns, {number: [item] for number, item in selected.items()},
                                image, min_evidence=0.30)
        relaxed_margin = (relaxed[0].score-relaxed[1].score if len(relaxed) >= 2
                          else float("inf") if relaxed else 0.0)
        if relaxed and (chosen_pattern is None or
                        relaxed[0].name == chosen_pattern.name or
                        (relaxed[0].matched >= 4 and relaxed_margin >= 6.0)):
            chosen_pattern = relaxed[0]
            ranked = relaxed
            reranking_after_backfill = True
    subpixel_shifts: dict[str, list[float]] = {}
    for number, candidate in list(selected.items()):
        patch = cv2.imread(str(folder / "patches" / f"patch_{number:02}.png"), 0)
        refined = refine_center(image, patch, candidate)
        selected[number] = refined
        subpixel_shifts[f"patch_{number:02}"] = [refined.x-candidate.x,
                                                 refined.y-candidate.y]
    result = row.copy()
    tree = cKDTree(chosen_pattern.projected) if chosen_pattern else None
    for number in range(1, int(row["n_patches"]) + 1):
        choice = selected.get(number)
        membership = int(choice is not None and tree is not None and
                         tree.query([choice.x, choice.y])[0] <= 24)
        result[f"patch_{number:02}"] = format_cell(
            (round(choice.x), round(choice.y), membership) if choice is not None else None)
    result["constellation"] = chosen_pattern.name if chosen_pattern else "unknown"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / f"{scene}.json").write_text(json.dumps({
        "scene": scene,
        "effective_presence_threshold": threshold,
        "naming_fallback": naming_fallback,
        "constellation_ranking": [item.serializable() for item in ranked[:8]],
        "geometric_changes": changes,
        "hypothetical_changes": hypothetical_changes,
        "presence_backfill": presence_backfill,
        "reranking_after_backfill": reranking_after_backfill,
        "subpixel_shifts": subpixel_shifts,
        "selected": {f"patch_{number:02}": item.serializable()
                     for number, item in selected.items()},
        "probabilities": {f"patch_{number:02}": probabilities.tolist()
                          for number, (_, probabilities) in options.items()},
    }, indent=2))
    _render_overlay(image, selected, chosen_pattern,
                    diagnostics / f"{scene}_overlay.png")
    return result
