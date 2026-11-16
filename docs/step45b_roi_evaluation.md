# Step 4.5B: reviewed full-image / ROI evaluation

This evaluates saved predictions only, without loading a detector or training.
Executed November 16, 2026 after scope validation passed for all eight images.

## Commands

```powershell
python scripts/validate_evaluation_scope.py --require-reviewed
python -B -m pytest -q -p no:cacheprovider
python -B scripts/evaluate_yolo.py
```

Actual test result: **96 passed**. The evaluator refuses existing output
directories. For a later authorized rerun, use a fresh --output-dir below
data/interim/yolo_ground_truth/evaluation/. No input files are changed.

## Rules

- Read actual classes.txt and require 0=car, 1=motorcycle, 2=bus, 3=truck.
  Map saved COCO IDs 2,3,5,7 to 0,1,2,3 explicitly.
- Require exactly eight mapped images and labels; reviewed scopes must pass the
  existing validator. Verify image hashes against originals and diagnostic
  provenance, configuration coverage, dimensions, counts and prediction schema.
- Convert YOLO normalized center/width/height annotations to pixel xyxy.
  Allow 1e-6 normalized rounding tolerance at annotation edges; do not clip.
- Include GT and predictions whose normalized centers are inside or on the
  reviewed rectangle boundary. Keep complete original boxes for IoU.
- Match within each image/configuration, class-aware at IoU >=0.5. Greedily accept
  candidate pairs in descending IoU, then GT index, then prediction index.
  Each GT/prediction can match once. This is not a global assignment optimizer.
- Unmatched predictions, including duplicates, are FP; unmatched GT are FN.
- Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN).
  A zero denominator yields JSON null / blank CSV. Thus F1=0 when errors exist
  without matches, and is undefined when all counts are zero.
- Signed count error=predicted_count-gt_count; count MAE is mean absolute error
  over eight images, not an average over classes.

## New experimental outputs

Under data/interim/yolo_ground_truth/evaluation/:

| File | Grain / contents |
| --- | --- |
| metrics.csv | Configuration and class, plus micro row; support/prediction counts, TP/FP/FN, precision/recall/F1 |
| per_image.csv | Image/configuration, scope and normalized bounds, before/after-scope counts, metrics and signed/absolute count error |
| per_image_class.csv | Image/configuration/class, scope, support and detection metrics |
| matches.csv | TP/FP/FN outcomes linked to image/configuration/class; zero-based parsed GT box index and original prediction CSV row (header is row 1); TP IoU |
| summary.json | roi-evaluation-1.0.0, rules, class mappings, aggregate metrics, count MAE, input/code hashes and limitations |

The tables are diagnostic artifacts, not approved production interfaces.
Camera IDs are represented by the unchanged original image path/filename;
no timestamp or timezone conversion is performed.

## Actual results

Five full scopes and three ROI scopes included 76 boxes (42 cars, 34 motorcycles)
from 101 annotations. Both annotated buses lie outside the reviewed ROI centers;
there are no annotated trucks. Bus/truck recall therefore has zero support.

| Configuration | TP | FP | FN | Precision | Recall | F1 | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 18 | 7 | 58 | .720000 | .236842 | .356436 | 6.375 |
| low_confidence | 29 | 104 | 47 | .218045 | .381579 | .277512 | 13.625 |

The lower threshold increases matched objects but also unmatched predictions.
It is not established as a better production threshold. Camera 1028's additional
motorcycle predictions still do not match its GT motorcycle at IoU >=0.5.

These eight images were hand-picked for diagnostic issues, not drawn as a
representative holdout. Results describe manually reviewed regions with mixed
full/ROI scopes. Scope declarations are trusted for completeness; the separate
review_notes.md still contains preparation-era pending text. Classification
ambiguity, box conventions and ROI-border sensitivity remain limitations.
No AP/mAP, confidence intervals or traffic-flow claims are made.
