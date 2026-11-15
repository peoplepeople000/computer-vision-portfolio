# Experiments and decisions

Jim Lo implemented the submitted Q1–Q3 notebook. Ta-Chun Tai implemented the selected Q4 competition version. Jim's competition experiments explored pose, keeper motion and temporal features. The competition comparisons below describe the final team report.

## Q1–Q3 notebook

**CNN interpretation.** Grad-CAM explains the predicted ImageNet class. The notebook compares ResNet50 and DenseNet121 at multiple depths and discusses why the portrait image's predicted class affects interpretation.

**Vehicle motion.** Restricting optical flow to instance masks reduces background contamination. Tracking IDs and apparent speed are inspectable in the video and per-frame measurements. Perspective and box jitter remain limitations, and road speed would require physical calibration.

**Occlusion.** A temporally consistent ID can still follow the wrong target. The notebook documents dog boxes expanding onto the person, rejects unreliable proposals and labels predicted positions during occlusion. The revision rejected 94 dog proposals, with predicted boxes at frames 330 and 340 and a detection at frame 360. This is a useful example of auditing image evidence behind a track.

[Notebook, results and occlusion analysis](notebooks/vision.ipynb)

## Q4 penalty-kick prediction

**What does the sequence add?** The final model combines geometry and final-frame RGB with five ordered SAM body tokens. On the first held-out source split, repeating the last token across all slots gave 70.99% fused kick accuracy. Using five observed tokens gave 72.52%, supporting the value of temporal information in that comparison.

| Held-out source split | Geometry + RGB kick accuracy | With SAM sequence |
| --- | ---: | ---: |
| Fold 1 | 70.61% | 72.52% |
| Fold 2 | 67.32% | 70.43% |

On fold 2, fusion corrected 14 kicks and introduced six errors. Both variants recognized only one of 37 middle kicks. The paired interval across 27 sources for the kick-accuracy gain was −0.35 to +5.75 percentage points. Repeated development use of these splits limits the precision of the estimate.

**How does missing evidence behave?** Missing RGB falls back to geometry. An ineligible SAM sequence falls back to the base prediction. Of 900 submission clips, 856 used SAM and 44 used the base branch. The goalkeeper has a static-pose fallback when motion observations are insufficient.

The team public score was 0.81777 under the competition's 70% kick / 30% keeper metric. CPU replay verifies the 900 submitted predictions from saved numeric features. The original videos and upstream models are needed for fresh feature extraction.

Figures are transcribed from the final submitted Q4 report, Sections 3.1 and 4.2, and the notebook's occlusion discussion. The private final-deliverable archive preserves both.

[Inference and fallbacks](competition/program/run.py) · [Recorded public score](competition/output/score_receipt.json)
