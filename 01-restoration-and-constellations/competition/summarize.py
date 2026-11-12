"""Summarize submission coverage and constellation-ranking uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nocpt.data import parse_cell, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, nargs="?",
                        default=Path("outputs/submission.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/scene_summary.csv"))
    parser.add_argument("--diagnostics", type=Path, default=None,
                        help="Directory containing per-scene pipeline result JSON files")
    args = parser.parse_args()
    _, rows = read_csv(args.submission)
    source = args.submission
    manifest_path = source.with_suffix(".manifest.json")
    if args.diagnostics is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if "base_csv_path" in manifest:
            source = Path(manifest["base_csv_path"])
    diagnostics_root = args.diagnostics or (
        source.parent / f"{source.stem}_diagnostics" / "results")
    audit_path = args.submission.with_suffix(".audit.csv")
    audit = {}
    if audit_path.exists():
        with audit_path.open(newline="") as handle:
            audit = {item["scene"]: item for item in csv.DictReader(handle)}
    records = []
    for row in rows:
        scene = row["Id"]
        n = int(row["n_patches"])
        present = sum(parse_cell(row[f"patch_{number:02d}"]) is not None
                      for number in range(1, n + 1))
        diagnostics_path = diagnostics_root / f"{scene}.json"
        diagnostics = json.loads(diagnostics_path.read_text())
        ranking = diagnostics["constellation_ranking"]
        guide = audit.get(scene, {})
        primary_name = guide.get("primary_name", row["constellation"])
        selected_rank = next((index+1 for index, candidate in enumerate(ranking)
                              if candidate["name"] == primary_name), None)
        records.append({
            "scene": scene,
            "predicted_constellation": row["constellation"],
            "primary_constellation": primary_name,
            "primary_name_rank": selected_rank if selected_rank is not None else "",
            "primary_top_pattern_score": round(ranking[0]["score"], 4) if ranking else "",
            "primary_top_margin": round(ranking[0]["score"]-ranking[1]["score"], 4)
                          if len(ranking) > 1 else "",
            "guided_score": guide.get("guided_score", ""),
            "guided_margin": guide.get("guided_margin", ""),
            "guided_support": guide.get("guided_support", ""),
            "guided_switched": guide.get("switched", ""),
            "present_patches": present,
            "total_patches": n,
            "present_fraction": round(present/n, 4),
            "backfilled_patches": len(diagnostics["presence_backfill"]),
            "geometry_changes": len(diagnostics["geometric_changes"]),
            "naming_fallback": diagnostics["naming_fallback"],
            "reranking_after_backfill": diagnostics["reranking_after_backfill"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} scene summaries to {args.output}")


if __name__ == "__main__":
    main()
