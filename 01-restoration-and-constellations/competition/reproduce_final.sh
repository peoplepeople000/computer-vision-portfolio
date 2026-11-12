#!/usr/bin/env bash
set -euo pipefail

# Run from the project root. The supplied polar CSVs are image-derived caches;
# generate_polar_cache.py recomputes image-only candidates for provenance checks.
python3 -m unittest discover -s tests -v
python3 train_model.py --union --model models/union/candidate.json
python3 cross_validate.py --union --threshold 0.75 --min-present-fraction 0.55 \
  --output outputs/experiments/union_cv_robust.csv
python3 evaluate.py outputs/experiments/union_cv_robust.csv
python3 run.py --union --model models/union/candidate.json \
  --config config_union.json --output outputs/submission_union_robust.csv
python3 experiments_guided.py --split validation --patch-pairs 64 --exact 40 --force
python3 apply_guided_names.py --base outputs/submission_union_robust.csv \
  --output outputs/submission.csv
python3 validate_submission.py outputs/submission.csv
python3 summarize.py outputs/submission.csv --output outputs/scene_summary.csv
