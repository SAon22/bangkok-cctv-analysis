# Step 4.5A: minimal YOLO diagnostics tooling

Tooling is implemented; **no real diagnostic inference has been executed**.
The exact image list must be confirmed by the user before running inference.

## Explicit input list

Edit `configs/yolo_diagnostic_images.json`. It accepts 1-8 unique project-relative
paths that exactly match the existing manifest. Each entry supports:
`image_path` (required), `note` (optional string), `nms` (optional boolean,
default false). Missing/unknown paths fail validation; no replacements are chosen.

The draft contains only camera 1028 at 2026-08-30 13:15:54, whose exact path
was found in the manifest. It is a review candidate, not confirmed ground truth.
No other examples have been selected automatically. Add the user's chosen
suspected misses, suspected spurious predictions and controls before confirmation.

Validate without loading weights, decoding images or running inference:

```powershell
python scripts/diagnose_yolo.py --images configs/yolo_diagnostic_images.json
```

After the user confirms the exact list, execute:

```powershell
python scripts/diagnose_yolo.py --images configs/yolo_diagnostic_images.json --run --weights models/yolov8n.pt --output-dir data/interim/yolo_diagnostics/step45a_01
```

The directory must be new, even if an earlier directory is empty. There is no
overwrite flag. No downloads occur; local weights must match the SHA-256 from
the saved feasibility summary. Existing feasibility outputs are only read.

## Configurations and outputs

All configurations use the same local model, CPU, image size 640, maximum 300
detections, class-aware NMS and rectangular letterboxing without augmentation.

| Configuration | Confidence | Classes | NMS IoU |
| --- | ---: | --- | ---: |
| baseline | 0.25 | 2, 3, 5, 7 | 0.7 |
| low_confidence | 0.05 | 2, 3, 5, 7 | 0.7 |
| all_classes | 0.05 | All COCO classes | 0.7 |
| nms_diagnostic | 0.05 | 2, 3, 5, 7 | 0.95 |

The last configuration runs only for entries explicitly marked `"nms": true`.

Each image gets one side-by-side PNG: original, baseline, low-confidence,
all-class, and optional NMS panel. Image regions retain original pixel dimensions;
a 45-pixel title strip is added above each region. EXIF orientation is normalized
in memory before prediction and drawing. Raw files are never changed.

Experimental outputs, separate from all production contracts:

- `detections.csv`: one post-NMS prediction per image/configuration; columns
  image_path, camera_id, timestamp, configuration, class_id, class_name, confidence,
  x_min, y_min, x_max, y_max. Coordinates are original oriented-image pixel xyxy.
  CSV has no stored type metadata: load camera_id as string and timestamps as
  timezone-naive datetimes. Empty outputs still have headers.
- `image_results.csv`: one attempted image/configuration, including successful
  zero-prediction and failed cases. Fields include identity, configuration,
  status, detection_count and error; decoded cases also have width, height and
  runtime_seconds. Failed counts are blank, not zero. Unreadable originals
  produce error rows without a fabricated comparison image.
- `provenance.json`: diagnostics-0.1.0 version, exact list/notes/NMS flags,
  comparison filenames, configurations, original-image hashes for decoded
  cases, model/manifest/script hashes, Python/package versions, timing and errors.

These are predictions, not TP/FP/FN labels or accuracy measurements. All-class
results are still post-NMS best-class predictions; no pre-NMS inspection is done.
The 300-detection cap remains in place and should be considered if a diagnostic
result reaches it. Low thresholds and high NMS IoU are diagnostic settings,
not proposed production defaults.
