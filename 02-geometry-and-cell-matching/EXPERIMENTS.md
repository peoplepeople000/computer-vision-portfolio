# Experiments and decisions

Jim Lo implemented the submitted Q5 competition pipeline. The team selected Ta-Chun Tai's Q1–Q4 notebook.

## Q1–Q4 notebook

The notebook uses line detection for runway boundaries, SIFT and RANSAC for a cover homography, pairwise correspondences for panorama stitching, and click-seeded GrabCut for segmentation. Jim's working notebook explored SAM for the segmentation task. In an interview, the submitted notebook is the reference for explaining the final algorithm and its outputs.

[Submitted notebook](notebooks/vision.ipynb)

## Q5 cell segmentation and matching

**Why inspect segmentation before pairing?** A geometrically plausible pair earns credit only when both endpoint masks meet the segmentation requirement. The final held-mouse evaluation had IV panoptic quality 0.65782 and EX panoptic quality 0.40225, identifying EX segmentation as a continuing weakness.

**Why compare pairing with fixed masks?** Keeping native instances fixed makes it easier to assess changes to candidate scoring. The final comparison used the same segmentation and varied the quality heads and score aggregation.

| Pair-scoring variant | Local composite | Public composite |
| --- | ---: | ---: |
| Original scorer | 0.52398 | 0.42791 |
| Upstream-unseen joint head alone | 0.52180 | Unsubmitted |
| Equal mean of both scores, selected final | 0.52545 | 0.42847 |
| Subsequent source aggregation | 0.52656 | 0.42832 |

The source-aggregation variant improved the local score but lowered the public score. The selected equal-weight ensemble retained all 8,360 IV masks and 2,796 EX masks while changing the pair set to 386 pairs.

**How was assignment constrained?** Candidate endpoints are mapped to native instances. Candidates above the fixed threshold are considered in descending score order, and each instance can be used once. Final pair selection uses deterministic greedy assignment. Hungarian matching appears earlier in contour association.

The small three-mouse cohort, repeated development use and inherited calibration limit interpretation of the small public improvement. Local and public metrics evaluate different data. These comparisons provide no statistical-significance claim.

Values and decisions are transcribed from the final submitted Q5 report, Sections 3.1 and 4.2. The private final-deliverable archive preserves the report and its complete model package.

[Frozen inference pipeline](competition/raw_engine/predict.py) · [Archived 29-region audit](competition/evidence/raw_inference/full29_parity_audit.json)
