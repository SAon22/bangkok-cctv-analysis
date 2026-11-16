# Step 6: detection data cleaning and validation

This step validates existing Step 5 intermediates. It does not run inference,
read raw image pixels, alter predictions, or implement Step 7 contracts.
No cleaning was necessary in the verified run: no rows were dropped, corrected,
clipped, deduplicated or assigned fabricated counts.

## Execute

From the project root, using existing dependencies:

```powershell
python -B scripts/validate_detections.py --run-dir data/interim/detection_runs/yolov8n_baseline_full_01 --manifest data/interim/image_manifest.parquet --output-dir data/interim/detection_validation
```

Relative paths resolve against the script's project root, with an optional
`--project-root` override. Existing report directories are refused. For a later
validation, supply a fresh `--output-dir` under `data/interim/`. Reports cannot
be written inside the input run or to a directory containing it.
Exit codes: 0 for structural success (warnings allowed), 2 for critical findings,
1 for an input/operational error preventing completion. Do not interpret a missing
report after an operational error as a successful validation.

## Checks and implementation

`engineering/detection_validation.py` reuses the existing Step 5 Arrow schemas,
version, class mapping, baseline, SHA-256 helper and manifest identity validator.
It leaves the existing `engineering/validate.py` placeholder API unchanged.
The resume reader is not called because its row-by-row recovery logic would
reread the full tables and does not provide category-level validation reports.

Each committed Parquet table is parsed once. A separate byte pass verifies its
committed checksum. Tables are concatenated in checkpoint order; pandas joins,
grouping and vectorized masks check the 274,096 prediction rows. Python loops
are limited to checkpoint files, a fixed set of fields/classes/checks, issue
formatting, and SHA-256 of image-path strings. No raw files are rehashed.

Critical checks cover:

- Exact Arrow schemas, compatible naive manifest timestamps/string camera IDs,
  run/version/baseline/manifest fingerprint and provenance hash formats.
- Contiguous committed ranges, batch sizes, positive finite durations,
  checksum/row-count consistency, no pending batches, and final summary agreement.
- Complete manifest coverage, unique image/detection keys, deterministic source
  indices/order, portable paths and path-derived image IDs.
- Required/nullable fields, allowed finalized statuses, failure explanations,
  success counts (including zeros), null failure counts and stage-specific fields.
- Parent references, unchanged camera IDs/timestamps, supported COCO IDs/names,
  contiguous detection indices and per-class/total count consistency.
- Finite confidence in [0,1] meeting the saved threshold; finite positive-area
  boxes inside the recorded decoded-image dimensions.
- Recorded source hash formats, availability after decoding, and consistency
  for repeated paths. Recorded dimensions and image hashes are not independently
  verified against raw pixels in this step.

Warning checks expose inference failures, missing core/optional metadata,
counts above an explicit review threshold of 100 per snapshot, and a class
representing over 90% of predictions. These thresholds are review prompts,
not evidence of incorrect predictions. Known model limitations and unresolved
timestamp provenance always remain warnings. An unusual prediction alone is
never removed or declared invalid.

## Report artifacts (detection-validation-1.0.0)

All artifacts are under `data/interim/detection_validation/`:

- `data_quality_report.json`: statistics, per-table null counts, per-category
  critical/warning issue counts, individual check results, source/run/model
  provenance, validator hash, elapsed validation time and Step 7 recommendation.
- `data_quality_report.md`: concise grouped checks, totals and limitations.
- `validation_issues.parquet`: one record per rule violation on a row, or one
  record for a dataset/checkpoint-level finding. The same row can violate
  multiple rules; issue counts are not counts of distinct bad images.

Issue fields: `severity`, `category`, `table`, `message` (Arrow string);
`row_index`, `detection_index` (nullable int64); `image_id`, `image_path`
(nullable string). Row indices are zero-based indices in concatenated image or
detection tables, or the loaded manifest. Dataset/checkpoint findings have null
row references. Prediction keys and all Step 5 schemas remain unchanged.
A failed warning check does not fail structural readiness; critical findings do.

Critical findings block Step 7 and identify investigation targets. Proposed
corrections require explicit review; the validator never applies corrections.

## Verified full-run execution

Executed the command above against run `15c14accabe04a87a03b02073ae332e3`.
Actual validation time: **8.5006 seconds**, excluding report writes.
**3,124 checks**, **0 critical issues**, **3 dataset-level warnings**.

| Result | Verified value |
| --- | ---: |
| Manifest/image rows | 61,319 / 61,319 |
| Cameras | 100 |
| Detection rows | 274,096 |
| Missing/unexpected images | 0 / 0 |
| Duplicate image/detection key rows | 0 / 0 |
| Successful/failed images | 61,319 / 0 |
| Successful zero-detection images | 10,488 |
| Committed checkpoints | 614 |
| Maximum snapshot prediction count | 32 |
| Car/motorcycle/bus/truck predictions | 208,919 / 12,592 / 29,161 / 23,424 |

Warnings: optional `location_en` is null in 614 manifest rows and
`location_detail_en` in 1,228 rows; baseline predictions have known accuracy
limitations; capture timezone/clock provenance is unverified. Null
`error_message` in all 61,319 successful rows is correct, not a missing-data error.
All core metadata and required result fields are populated.

**Ready for Step 7 contract planning on structural grounds.** Timezone evidence,
consumer semantics, schema approval and accuracy limitations remain explicit
integration decisions. This is not a claim of reliable vehicle recognition or
approval of the proposed UTC-aware production contract. Snapshot predictions
are not traffic flow, unique vehicles or congestion.

## Tests

Executed `python -B -m pytest -q -p no:cacheprovider` with third-party pytest
plugin autoload disabled: **153 passed in 12.35 seconds**, including 39 new
synthetic validation cases. Tests cover valid/zero/error records, missing and
duplicate keys, invalid classes/confidence/boxes, metadata/count mismatches,
hash conflicts, nullable timestamps, corrupt/incompatible checkpoints, stale
summaries, output overwrite refusal and preservation of input bytes.
Syntax compilation passed. Ruff was unavailable (`No module named ruff`);
no dependencies were installed. No YOLO inference, raw-image decoding, training,
Step 7 implementation, commit or push was performed.
