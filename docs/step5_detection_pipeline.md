# Step 5: checkpointed raw vehicle detection

Implemented raw engineering inference only. Step 6 cleaning/validation and Step 7
downstream contract/integration remain separate. The proposed UTC production
contract is not implemented: capture timestamps remain exactly as in the manifest.

## Run and resume

Use the existing environment and trusted local models/yolov8n.pt. No automatic
installation or weight download is implemented. Where available, the saved
feasibility weight hash must match the supplied file.

A limited run:

```powershell
python scripts/run_engineering.py --output-dir data/interim/detection_runs/trial_01 --batch-size 100 --limit 20
```

For full-dataset execution, omit --limit and
use a NEW directory:

```powershell
python scripts/run_engineering.py --output-dir data/interim/detection_runs/baseline_01 --batch-size 100
```

Resume with identical manifest, weights, batch size, image limit, code and package
versions, adding --resume. Config defaults are in configs/config.yaml.
Options --manifest, --weights, --project-root and --config follow repository path
conventions. Relative paths resolve from the project root.

--batch-size is the number of image records per durable checkpoint. Inference
remains serial using one reused detector instance; it is not a tensor minibatch.
--max-batches N stops cleanly after N new checkpoints and can change between
invocations. It does not change the target plan or compatibility signature.
A limited run cannot be extended to the full dataset by changing its limit on
resume: create a new run instead.

Baseline is fixed: YOLOv8n, confidence .25, image size 640, NMS IoU .7, CPU,
COCO IDs 2/3/5/7, max detections 300, augmentation disabled. The existing detector
and path/manifest validators are reused, with no model tuning or training.
Original images are decoded one at a time and EXIF-normalized in memory.

## Files and checkpoints

```text
<run>/
  run.json
  run.lock
  summary.json
  batches/
    batch-00000000-00000100/
      image_counts.parquet
      detections.parquet
      commit.json
```

run.json is immutable run identity/configuration/provenance. Its signature covers
the exact manifest SHA-256, weights SHA-256, baseline, batch size, limit, package/
Python versions and the pipeline/checkpoint/detector/reused-validator source hashes.
The plan sorts by string camera_id, timestamp (null last), then image_path.

Each batch is written into a unique .pending-* directory. Both Parquet files
and commit JSON are flushed/fsynced, then the directory is atomically renamed
on the same filesystem. The committed directory is the transaction boundary:
image and detection files become visible together. The summary is a replaceable
cache, rebuilt from committed files; its absence/staleness does not lose progress.

Recovery verifies checksums, schemas, contiguous source indices, source identity
and detection keys before continuing. Interrupted uncommitted batches may be
inferred again, but published records are not appended twice. Orphan .pending-*
directories are ignored and retained, not automatically deleted. An OS file
lock prevents simultaneous writers and releases on process exit/crash.

Checkpoint corruption or incompatible resume is an error, not skipped data.
A run missing run.json cannot be safely identified; use a fresh directory.
This protects normal process interruption on a local filesystem. It is not a
backup or a guarantee against disk failure or all power-loss/filesystem behaviors.
Raw images must remain immutable; per-image content hashes are recorded, but
resume does not rehash the entire source image collection.

Read committed tables explicitly (do not read the mixed batches directory as
one generic Parquet dataset):

```python
parts = sorted((run_dir / "batches").glob("batch-*"))
image_counts = pd.concat(
    [pd.read_parquet(p / "image_counts.parquet") for p in parts],
    ignore_index=True,
)
detections = pd.concat(
    [pd.read_parquet(p / "detections.parquet") for p in parts],
    ignore_index=True,
)
```

## Raw engineering schema: detection-raw-1.0.0

All parts use explicit Arrow schemas, including empty detection tables.
These are internal intermediate outputs, not an approval of Step 7 contracts.

Shared fields in both tables:

| Fields | Arrow type | Semantics |
| --- | --- | --- |
| schema_version, run_id | string | Version and one run UUID |
| image_id | string | SHA-256 of UTF-8 relative image_path; identity, not content |
| image_path | string | Portable project-relative source path |
| camera_id | string | Unchanged manifest ID, preserving leading zeros |
| timestamp | timestamp[us], nullable, no timezone | Unchanged filename clock time from manifest |

image_counts.parquet grain: one attempted image. Key: (run_id, image_id).
Additional fields:

| Fields | Arrow type | Semantics |
| --- | --- | --- |
| source_index | int64 | Zero-based position in the deterministic limited plan |
| status | string | success, invalid_path, missing_file, unreadable, inference_error |
| error_message | nullable string | Failure explanation; null on success |
| source_sha256 | nullable string | Original file-content hash when bytes were read |
| width, height | nullable int64 | Original EXIF-oriented decoded dimensions |
| car_count, motorcycle_count, bus_count, truck_count | nullable int64 | Predicted per-class counts; null on failures |
| total_vehicle_count | nullable int64 | Sum of the four counts; null on failures |
| runtime_seconds | double | Per-image path/read/decode/inference elapsed time |
| inference_seconds | nullable double | Predict-call time, including failed calls |
| processed_at | timestamp[us, UTC] | Processing time, separate from capture timestamp |

detections.parquet grain: one retained prediction.
Key: (run_id, image_id, detection_index); foreign key references its image row.

| Fields | Arrow type | Semantics |
| --- | --- | --- |
| detection_index | int64 | Zero-based prediction ordinal within the image |
| class_id | int64 | COCO 2=car, 3=motorcycle, 5=bus, 7=truck |
| vehicle_class | string | Corresponding COCO class name |
| confidence | double | Model score, at least .25 |
| x_min, y_min, x_max, y_max | double | Source-oriented pixel xyxy coordinates |

Successful zero predictions produce a complete image row with five zero counts
and no detection rows. Failed images have null counts and no detection rows;
they are never represented as zero vehicles. Committed failure rows are skipped
on resume just like successful rows. Retrying them requires a separate run.
Fatal model setup or checkpoint failures stop the run rather than fabricating
per-image inference failures. A complete summary means every target was attempted,
not that every image succeeded; inspect status_counts.

## Verified tests and real smoke run

All **114 tests passed**, including mocked errors/zeros, before-/after-publication
interruption, recovery, duplicate prevention, corrupt/incompatible checkpoints,
completed no-op resume and OS locking. No dependencies were installed.

A new seed-42 smoke manifest contains 20 images from 20 cameras, all seven dates,
and all three clock bands (14 morning, 4 midday, 2 evening). No original manifest
or evaluation artifact was modified.

Actual commands:

```powershell
python -B scripts/run_engineering.py --manifest data/interim/detection_pipeline_smoke_manifest.parquet --output-dir data/interim/detection_pipeline_smoke --batch-size 5 --limit 20 --max-batches 2
python -B scripts/run_engineering.py --manifest data/interim/detection_pipeline_smoke_manifest.parquet --output-dir data/interim/detection_pipeline_smoke --batch-size 5 --limit 20 --resume
```

The first invocation committed 10 images; the second committed the remaining 10.
Four checkpoint pairs contain 20 unique image rows and 98 unique detections:
60 cars, 3 motorcycles, 23 buses and 12 trucks. All 20 succeeded; three had zero
detections. The six files from the first two checkpoints remained byte-identical.
A completed resume with model construction explicitly disabled succeeded without
any further inference. Image identities, count sums, camera IDs, timestamps and
all source-content hashes were checked.

Example successful zero: camera 1076, 2026-08-25 06:21:11, total_vehicle_count=0.
Example nonzero: camera 11, 2026-08-31 07:20:07, car=10, motorcycle=0, bus=3,
truck=1, total=14. One bus prediction has confidence .74472 and
xyxy=(0.00,110.99,56.43,181.55). These are predictions, not ground truth.

## Runtime estimate and remaining gates

The two-invocation smoke run recorded 5.388 seconds in predict calls and 5.470
seconds of image processing, including two cold starts. Excluding the first image
of each invocation, the 18-image steady mean was 34.55 ms predict / 37.55 ms image
processing, approximately 28.9 / 26.6 images per second.

A linear extrapolation to 61,319 images is about 35.3 minutes of predict calls
or 38.4 minutes of image processing, plus startup, checkpoint/fsync I/O, recovery
verification and hardware/workload variation. This small biased smoke sample is
not a benchmark or guaranteed completion time.

Before full execution: authorize a fresh full-run output directory, check local
disk capacity, retain immutable inputs/weights/software, and agree that this
baseline's known detection limitations are acceptable for raw prediction
collection. The eight-image diagnostic evaluation showed poor recall and
threshold-sensitive false/duplicate predictions; it was not representative
holdout evaluation. Do not interpret counts as traffic flow/congestion.
Steps 6 and 7, timezone verification and downstream integration remain future work.

## Completed full-dataset execution

The full dataset has now been processed with the existing baseline and pipeline.
This supersedes the earlier smoke-only milestone; no Step 6 or Step 7 work began.

Actual command:

```powershell
python -B scripts/run_engineering.py --manifest data/interim/image_manifest.parquet --weights models/yolov8n.pt --output-dir data/interim/detection_runs/yolov8n_baseline_full_01 --batch-size 100
```

Preflight checked exactly 61,319 unique manifest paths, opened all 61,319 raw
files successfully and found them nonempty, and verified baseline/weights/code/
package compatibility with the smoke run. Free space was 28.42 GiB; smoke-scaled
output estimate was 0.193 GiB, with an 18.54 GiB conservative reserve check.
The target directory was fresh. No image limit or batch-count limit was applied.

| Result | Actual value |
| --- | ---: |
| Processed / successful images | 61,319 / 61,319 |
| Failed images | 0 |
| Successful zero-detection images | 10,488 |
| Unique cameras | 100 |
| Detections | 274,096 |
| Car / motorcycle / bus / truck | 208,919 / 12,592 / 29,161 / 23,424 |
| Unique image / detection keys | 61,319 / 274,096 |
| Duplicate image / detection keys | 0 / 0 |
| Missing manifest images | 0 |
| Committed / pending checkpoints | 614 / 0 |
| Process wall time | 3,691.060 seconds (1 h 1 min 31 s) |

Run ID: 15c14accabe04a87a03b02073ae332e3.
The run completed in one invocation with exit code 0; no restart/resume was needed.
Every checkpoint checksum, schema, contiguous source index, key/reference,
manifest camera ID/timestamp and count total was verified after finalization.
All 215 protected smoke/evaluation/diagnostic/input/code files were unchanged.

Results: data/interim/detection_runs/yolov8n_baseline_full_01/.
The authoritative image and detection tables remain in the paired checkpoint
Parquet files. summary.json contains final totals. execution_report.json records
preflight, command, actual timing and audit results. failed_images.parquet is a
typed empty image-schema table because there were no failures; it is an
operational failure extract, not a cleaned dataset. The committed outputs were
about 33.3 MiB before the small audit sidecars.

The measured wall time exceeded the earlier small-sample extrapolation. These
are raw model predictions with known quality limitations, not verified vehicles,
traffic flow or congestion. No model tuning, retraining, package installation,
raw-data modification, cleaning, integration, commit or push was performed.
