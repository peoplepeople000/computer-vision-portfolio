# Experiments and decisions

The notebook and competition are separate parts of this project. Jim Lo's implementations were selected for both.

## Q1–Q4 notebook

For galaxy restoration, the notebook compares a raw mean, intensity-calibrated averaging and agreement-weighted fusion on matching crops. Alignment and intensity calibration address different errors. Weights reduce the influence of inconsistent observations, while shared blur and bias can remain. With no clean reference, the notebook documents visual evidence and calibration diagnostics. A useful interview discussion is how to evaluate a reconstruction when PSNR and SSIM cannot be computed against ground truth.

For template localization, the notebook adds high-frequency detail checks, spatial agreement and horizontal wrapping to handle confusing matches and the panorama seam. Q3 implements filtering, Sobel gradients, non-maximum suppression and hysteresis in NumPy.

[Notebook and comparisons](notebooks/vision.ipynb)

## Q5 constellation detection

**Why keep two retrieval methods?** In the final report's training-query diagnostic, polar retrieval found 62 of 71 present queries, while NCC found 52. On disagreements, polar was correct 15 times and NCC five times. This motivated retaining both candidate banks and verifying them with the same image-based features.

**Why use geometry after image matching?** Repeated star patterns can produce convincing local matches. Constellation geometry adds node support and coverage checks. The correction is restricted to independently retrieved candidates, and a common evidence gate controls changes to uncertain constellation names.

| Method | Evaluation protocol | Mean score |
| --- | --- | ---: |
| Wider NCC and verifier | Leave-one-scene-out, predicted name | 0.8698 |
| Full candidate union and geometric correction | Leave-one-scene-out, predicted name | 0.9425 |

These configurations differ in several components. The table supports the overall development choice but cannot isolate each component's effect. Other report ablations used oracle names or in-sample fitting and are omitted from this comparison.

The final public score was 0.86206. Only three labeled scenes supported local validation, and their repeated use for development limits generalization claims. Historical polar candidates are retained in the final package because exact replay depends on them.

These figures are transcribed from the final submitted Q5 report, Section 4.2. The private final-deliverable archive preserves that report.

[Inference and geometric checks](competition/run.py) · [Replay procedure](competition/reproduce_final.sh)
