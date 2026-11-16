# Step 4: YOLO feasibility testing

This is a bounded engineering experiment, not the full pipeline and not a
validated vehicle-count dataset. It reads the existing image manifest and
decodes selected images one at a time. Raw images are read only.

## Usage

Run from the project root using the environment containing the declared runtime
dependencies. No new project dependency or version pins were added.

Sampling-only needs pandas, PyArrow and Pillow but does not import Ultralytics,
load weights, or decode images:

```powershell
python scripts/test_yolo_feasibility.py --sampling-only --sample-size 100 --seed 42 --output-dir data/interim/yolo_feasibility_sampling_only
```

Actual inference prefers local weights:

```powershell
python scripts/test_yolo_feasibility.py --sample-size 100 --seed 42 --confidence 0.25 --weights models/yolov8n.pt --device cpu
```

If the default weights are missing, add `--download-weights` explicitly. It
downloads official YOLOv8 nano weights from the Ultralytics assets release into
models/yolov8n.pt. Other weight names must be supplied locally. Missing weights
without opt-in are an actionable model error, not an automatic download.
Only load trusted model weights.

The actual command executed for the verified run was:

```powershell
python -B scripts/test_yolo_feasibility.py --sample-size 100 --seed 42 --confidence 0.25 --weights models/yolov8n.pt --download-weights --device cpu
```

Defaults: sample size 100, seed 42, confidence .25, IoU .7, input size 640,
maximum detections per image 300, CPU. Use --help for all options, including
--project-root, --output-dir, --image-size and --max-detections. --dry-run aliases
--sampling-only. Paths resolve from the project root, not the shell directory.
Outputs must be below data/interim; nonempty run folders are refused. For repeat
experiments, choose a new --output-dir rather than overwriting evidence.

Expected artifacts in the inference output directory:

- sample_manifest.parquet: selected image identity, metadata, status, nullable
  prediction counts, dimensions, timing and annotation paths.
- detections.parquet: one row per vehicle prediction, typed even when empty.
- summary.json: provenance, coverage, counts, failures and runtime statistics.
- annotated_images/: separate JPEG copies, including successful zero-count images.

The detailed experimental contract is in [the dictionary](data_dictionary.md).
Production contracts remain proposed; downstream teams must not silently adopt
these experimental outputs as their final interface.

## Sampling and limitations

The sampler sorts paths, uses a fixed Python random seed to choose one candidate
per nonempty camera/date/clock-band stratum, shuffles each camera's candidates,
then selects round-robin across cameras. Clock bands are 00-10, 11-15 and 16-23.
Invalid timestamp rows remain eligible in an unknown stratum.

At most one image per stratum is selected. Oversized budgets return all available
strata with a warning. Small budgets cannot cover every combination; budgets
smaller than the number of cameras omit some cameras. This is camera-balanced
feasibility sampling, not an unbiased image census or an accuracy evaluation.
A fixed seed guarantees sampling reproducibility for unchanged inputs, not
bitwise prediction reproducibility across hardware/library versions.

Predictions use COCO IDs 2 (car), 3 (motorcycle), 5 (bus), and 7 (truck).
Vans/pickups follow the pretrained model's labels; no production taxonomy decision
is implied. Confidence scores are model scores, not empirical accuracy.
Successful zero detections do not establish an empty road. Snapshot counts are
not traffic flow, congestion, or unique passing vehicles.

Images are read with Pillow before inference to support Unicode paths and record
decode failures. Missing/unsafe paths, unreadable images, inference failures,
and annotation errors remain explicit. No failed inference is replaced by zero.
File existence and successful decoding do not establish visual suitability.

## Verified actual run

Inspected November 15, 2026:

| Item | Actual result |
| --- | ---: |
| Sampled images / cameras / dates | 100 / 100 / 7 |
| Clock-band sample counts 00-10 / 11-15 / 16-23 | 36 / 31 / 33 |
| Successful images | 100 |
| Successful zero-detection images | 14 |
| Failed images / annotation errors | 0 / 0 |
| Annotated copies | 100 |
| Total predictions | 444 |
| Car / motorcycle / bus / truck predictions | 318 / 12 / 75 / 39 |
| Mean / median predict-call time | 0.0649 / 0.0322 seconds |
| Model load including download | 8.312 seconds |
| Elapsed before final output writes | 17.323 seconds |

Model: yolov8n.pt; SHA-256:
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36.
Runtime: Ultralytics 8.4.213, torch 2.15.0+cpu, torchvision 0.29.2+cpu.
These times include first-call warmup and are one local run, not benchmarks.
The sampling-only and real-run image IDs matched exactly. Foreign keys,
per-image counts, original naive timestamps and all 100 annotation paths were
checked after Parquet round-trip.

The selected interpreter initially lacked Ultralytics/PyTorch. Runtime installation
commands executed (without editing pyproject.toml):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics
```

Unit tests mocked inference and needed no weights. Existing tests plus new
sampling, schema, zero-output, path, inference/annotation-failure and CLI cases
passed before actual inference. See the final verification below for the final
test count.

No precision, recall, accuracy, mAP or class-quality claim is supported: no
annotated ground truth was used. Low-resolution scenes, small/occluded vehicles
and class confusion require manual review and a labelled evaluation set before
full-batch inference or downstream interpretation.

References: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/),
[prediction API](https://docs.ultralytics.com/modes/predict/).

## Final verification (November 15, 2026)

- `python -B -m pytest -q -p no:cacheprovider` completed with **53 passed**.
  PYTHONDONTWRITEBYTECODE=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 were set.
- The existing 100-image run was preserved and independently rechecked:
  source-manifest/weight checksums, deterministic sample identities, detection
  foreign keys, counts, timezone-naive timestamps and all 100 annotation files.
- A real single-image smoke check of the preserved
  `detect_vehicles(image_path, weights_path)` API matched its saved prediction
  counts after removing the deprecated half=False argument.
- `git diff --check` passed. Ruff was not installed, so lint was not claimed.
- No inference failures or annotation errors occurred in the saved run.
  The original run emitted a nonfatal Ultralytics half-argument deprecation
  warning; the argument was removed and the current API smoke check emitted none.
- The completed artifacts were not regenerated during final verification.
  Raw files, manifest, existing run outputs and unrelated work were preserved.
