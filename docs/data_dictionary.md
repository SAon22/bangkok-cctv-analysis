# Data dictionary and integration contracts

Inspected **November 15, 2026**, using current source code, the local CSV,
Parquet schema/values, JSON reports and saved notebook outputs. **EXISTING**
describes available artifacts. **PROPOSED v1.0.0** describes an unimplemented,
unapproved design. This documentation does not migrate any data or change code.

## Shared conventions

- Project-relative paths use forward slashes and are resolved against the
  repository root. Raw inputs are immutable.
- camera_id is a **string**, never an integer key. Preserve leading zeros.
  Current loaders trim surrounding whitespace; do not silently change the ID
  representation or assume filenames override directory IDs.
- Existing timestamp values are timezone-naive filename clock readings.
  Do not interpret them as UTC or automatically localize them to Asia/Bangkok.
- Image availability and visible vehicles in snapshots are different measures.
  Neither is a count of unique vehicles passing a road.
- CSV has no intrinsic types; the types below describe inspected pandas loading.
  Parquet physical types are reported separately. The original CSV/manifest have no explicit
  schema_version field or enforced database primary keys.

## EXISTING: raw image files

Input scan root: `data/raw/cctv_images/`; current local images reside under
`data/raw/cctv_images/cctv_images/<camera_id>/<filename>`.

The manifest builder scans recursively and accepts .jpg, .jpeg and .png,
case-insensitively. It takes camera_id from the **immediate parent directory**
and parses the final `_YYYYMMDD_HHMMSS.<extension>` suffix. Invalid/missing dates
become NaT rather than dropping the image. It does not compare the camera ID
prefix in the filename to the parent ID and does not decode images.

The exploration script instead enumerates numeric camera directories after
handling the optional nested cctv_images folder, and decodes up to five files.
These implementations are equivalent for the current layout, not necessarily
for arbitrary future folder nesting.

## EXISTING: data/raw/cctv_info.csv

Grain: one camera metadata record. Observed key: camera_id (100 unique, non-null
IDs in 100 rows). Read with `pd.read_csv(..., dtype={"camera_id": "string"})`.
Descriptions reflect source labels, not independently verified translations,
geolocation, network access, or capture-time provenance.

| Field | Loaded type | Meaning | Observed nulls |
| --- | --- | --- | ---: |
| camera_id | pandas string | Source camera identifier, join key | 0 |
| location | string | Source location label, commonly Thai | 0 |
| location_detail | string | Source detailed location description | 0 |
| location_en | string | Source English-labelled location text; translation quality unverified | 1 |
| location_detail_en | string | Source English-labelled detailed description | 2 |
| latitude | float64 | Source latitude in decimal degrees; CRS/datum unverified | 0 |
| longitude | float64 | Source longitude in decimal degrees; CRS/datum unverified | 0 |
| ip_address | string | Source camera network address; not a stable analytical join key | 0 |
| image_count | int64 | Expected available image files per camera; not vehicle counts | 0 |

pandas default CSV null parsing applies. Text has no custom missing-value
normalization beyond camera ID handling in the consuming code. The current
manifest join rejects null/blank or duplicate metadata camera IDs and requires
camera_id, location, latitude and longitude columns. It does not require all
metadata values to be non-null or establish their correctness.

## EXISTING: data/interim/image_manifest.parquet

Producer: `src/bangkok_cctv/engineering/image_manifest.py`, via
`scripts/build_image_manifest.py`.

Grain: **one row per discovered image file**; current shape: **61,319 x 12**.
Observed unique key: image_path. The current (camera_id, timestamp) pairs are
also unique, but that is not enforced by the builder and is not a safe universal
image key. No schema version column is present. Arrow storage fields are
nullable; observed non-null data is not a schema-level non-null constraint.

| Field | Arrow physical type | Meaning / origin | Current nulls |
| --- | --- | --- | ---: |
| camera_id | large_string | Immediate image parent directory name, trimmed string; CSV join key | 0 |
| timestamp | timestamp[us], no timezone | Final filename suffix parsed as calendar datetime | 0 |
| image_path | large_string | Project-root-relative POSIX path to original image | 0 |
| location | large_string | Joined CSV location | 0 |
| location_detail | large_string | Joined CSV detailed location | 0 |
| location_en | large_string | Joined English-labelled location | 614 |
| location_detail_en | large_string | Joined English-labelled detailed location | 1,228 |
| latitude | double | Joined source latitude | 0 |
| longitude | double | Joined source longitude | 0 |
| ip_address | large_string | Joined source network address | 0 |
| image_count | int64 | Camera-level expected file count, repeated for every image of that camera | 0 |
| metadata_matched | bool | True if the image camera ID matched a CSV record | 0 |

Current pandas timestamp dtype is datetime64[us]. String dtype spelling can vary
by pandas version; preserve its logical string semantics.

The join is left/many-to-one: unmatched images are retained with
metadata_matched=False and null joined values. All 61,319 current rows are
matched. Failed timestamp parsing is represented by NaT/Parquet null, not a
fabricated capture time. The current dataset has no such failures.

Do not sum image_count across manifest rows: that would multiply a camera-level
value by the number of images. Use row counts for image availability.

Current observed range: 2026-08-25 06:00:00 to 2026-08-31 21:01:19, seven dates,
100 cameras. Relative paths include the current nested cctv_images directory;
consumers should resolve stored paths rather than reconstructing filenames.

## EXISTING: data/interim/dataset_report.json

Producer: `scripts/explore_dataset.py`. Grain: one dataset exploration run.
No explicit schema version or run timestamp is recorded; the file is overwritten
by subsequent runs. Fields can be absent when a run stops early.

| JSON path | Type / semantics |
| --- | --- |
| status | string: complete or incomplete; complete means routine finished, not absence of all quality issues |
| sources.csv, sources.images | Project-relative string paths to inspected inputs |
| errors | Array of fatal error strings; [] in the current complete report |
| csv.shape | Two integers: rows, columns |
| csv.columns | Ordered array of column-name strings |
| csv.missing_values | Column-name to null-count integer map |
| csv.duplicate_camera_ids | Array of string IDs occurring more than once |
| csv.invalid_camera_id_rows | Array of zero-based DataFrame row indices with missing IDs |
| csv.invalid_image_count_rows | Array of zero-based row indices with missing/non-numeric/negative/non-integral counts |
| csv.image_count_statistics | count, mean, std, min, 25%, 50%, 75%, max; numeric or null |
| images.camera_count, images.image_count | Integer directory-camera count and supported-image file count |
| images.image_counts_by_camera | String camera ID to integer file-count map |
| images.ignored_file_count | Integer count of non-image files encountered under the script's scan rules; not an exhaustive count of excluded subtrees |
| images.sample_filenames | Up to ten project-relative image-path strings |
| comparison.cameras_missing_from_images | CSV camera IDs absent from camera directories |
| comparison.cameras_missing_from_csv | Directory camera IDs absent from CSV |
| comparison.image_count_mismatches | Array of objects: csv_row_index integer, camera_id string, expected integer, actual integer |
| sample_images | Up to five objects: path; width/height integers and format string when available; error string on failure |
| unreadable_sample_images | Subset of sample_images containing error |

Current report: complete; 100 cameras, 61,319 images, no camera/count
mismatches, five recorded readable samples and no fatal errors. Sample success
does not certify unsampled images. An incomplete report must not be treated as
a successful validation.

## EXISTING: data/interim/image_manifest_summary.json

Producer: manifest module. Grain: one manifest-build summary, overwritten when
the producer runs. No explicit version/run ID is stored.

| JSON field | Type / semantics |
| --- | --- |
| total_images, camera_count | Integer row count and distinct camera count |
| timestamp_timezone | String documenting naive filename time without conversion |
| date_range | Object with start/end ISO datetime strings, or null when no valid timestamps |
| unique_dates | Integer distinct valid calendar-date count |
| images_per_date | YYYY-MM-DD to integer image-count map, valid timestamps only |
| images_per_camera | String camera ID to integer row-count map, including invalid-time rows |
| sampling_interval_units | String: seconds |
| sampling_interval_policy | String describing sorted consecutive timestamps, zero duplicate intervals and cross-date gaps |
| sampling_interval_statistics_per_camera | Camera-ID map to count, mean, std, min, 25%, 50%, 75%, max |
| duplicate_timestamp_group_count | Integer number of camera/timestamp groups containing more than one image |
| duplicate_timestamp_image_count | Integer total rows in those groups, including each group's first row |
| duplicate_timestamp_excess_images | Integer sum of group size minus one |
| duplicate_timestamps | Array of camera_id string, timestamp ISO string and image_count integer |
| invalid_or_missing_timestamp_count | Integer number of null/invalid timestamps |
| invalid_or_missing_timestamp_paths | Array of affected project-relative image paths |
| unmatched_metadata_image_count | Integer rows with metadata_matched=False |

Interval statistics exclude each camera's first valid timestamp and all invalid
timestamps. They include overnight/multiday gaps and zero intervals from
duplicates. Statistics use pandas describe: count is serialized as a number
(often e.g. 610.0); undefined values become JSON null; std is sample standard
deviation. A camera with no intervals has count=0, undefined remaining statistics.
The 30-minute gap threshold is a notebook analysis, not a field in this report.

## EXISTING: notebook outputs (not a production contract)

`notebooks/01_data_exploration.ipynb` contains six sections, 10 executed code
cells, eight embedded PNG figures and no saved cell errors. It reads existing
artifacts and stores outputs within the notebook; it does not create a new
production table or rebuild the manifest.

The saved seed-42 sample selects one image per camera per observed clock band
(00-10, 11-15, 16-23); 300 files were sampled, 100 per band, with no recorded
decode failures. Recorded combinations: JPEG 352x288 (296), JPEG 348x261 (3),
PNG 400x266 (1). These are sample statistics, not dataset-wide resolution counts.

The notebook's >30-minute-gap rule reports 671 intervals: 71 within a date,
600 crossing dates. The broad coordinate review box is 13-14.5 degrees N and
100-101.5 degrees E, not an official boundary or proof of CRS accuracy.

## PROPOSED v1.0.0: future vehicle-count and analytics contracts

**NOT IMPLEMENTED. NOT APPROVED.** These proposals refine the original provisional
four-class count schema. None of the following output paths or fields are claims
about currently available datasets. Existing manifest/CSV schemas remain unchanged.
Engineering, Data Science and Dashboard must approve the contract before use.

### Versioning, identity and time proposal

- Use semantic versions. Persist schema_version="1.0.0" in every proposed output.
  A breaking change to grain, key, type, timezone, class meaning or missing-data
  semantics requires a major version and explicit consumer agreement. Additive
  nullable fields require a documented minor version; non-semantic clarifications
  may use a patch version.
- camera_id remains the trimmed string used by the manifest; preserve zeros.
- image_path references the existing manifest and remains project-relative.
  It identifies a file location, not immutable image content. Renames require
  a migration or a future approved content-ID extension.
- model_run_id is a non-null, unique run identifier referencing a run-provenance
  JSON record with model identifier, weight checksum, library versions,
  configuration/confidence threshold, class mapping and UTC execution time.
  No particular model is selected by this proposal.
- Preserve filename_timestamp as a naive microsecond timestamp. Propose
  timestamp as a UTC-aware microsecond timestamp only when source timezone
  evidence is confirmed. Store the confirmed IANA zone in source_timezone;
  Asia/Bangkok is a candidate, not an established fact.
- While the source zone is unknown, timestamp and source_timezone are null.
  Do not guess UTC instants. Such rows can carry image inference results but
  cannot enter time-bucketed analytics. Display converted timestamps using
  Asia/Bangkok only after time provenance is established.
- Invalid filename times remain null. No automatic imputation of times or
  interpolation of counts is proposed.

### PROPOSED image_vehicle_counts v1.0.0

Candidate location: `data/processed/image_vehicle_counts_v1.parquet`.
Grain: **one attempted/explicitly skipped image per model run**, not one vehicle.
Primary key: **(model_run_id, image_path)**. Simultaneous images do not collide
because timestamp is not the primary key. Inference should account explicitly
for every manifest image selected for the run.

| Intended field | Intended Arrow type | Required semantics |
| --- | --- | --- |
| schema_version | string | Non-null, 1.0.0 |
| model_run_id | string | Non-null run-provenance key |
| image_path | string | Non-null manifest image_path foreign key |
| camera_id | string | Non-null, equal to referenced manifest ID |
| filename_timestamp | timestamp[us], naive | Nullable; original parsed clock value |
| source_timezone | string | Nullable; verified IANA timezone, not a guessed value |
| timestamp | timestamp[us, UTC] | Nullable until clock/zone are verified and parsing succeeded |
| inference_status | string enum | Non-null: success, unreadable, inference_error, skipped |
| error_message | string | Null for success; non-empty error/skip explanation otherwise |
| car_count | int64 | Nonnegative for success; null otherwise |
| motorcycle_count | int64 | Nonnegative for success; null otherwise |
| bus_count | int64 | Nonnegative for success; null otherwise |
| truck_count | int64 | Nonnegative for success; null otherwise |
| total_vehicle_count | int64 | Sum of four class counts for success; null otherwise |
| processed_at | timestamp[us, UTC] | Non-null processing/skip-record time, distinct from capture time |

No per-object bounding-box table is implemented or required by this proposal.
If needed, design a separately versioned object-level contract with its own
detection ordinal/key rather than changing this table's row grain.

### PROPOSED vehicle taxonomy

| Class / count field | Intended meaning |
| --- | --- |
| car / car_count | Passenger cars under an approved model-to-project mapping; treatment of vans and pickups needs explicit agreement |
| motorcycle / motorcycle_count | Motorized two-wheel vehicles; bicycles are excluded |
| bus / bus_count | Passenger buses/coaches under the approved mapping |
| truck / truck_count | Goods trucks under the approved mapping; pickup boundary must be agreed |
| total_vehicle_count | Sum of the four disjoint mapped categories, not all visible road users |

Pedestrians and bicycles are excluded from these vehicle counts. Ambiguous or
unmapped labels must not be silently reassigned. Freeze class IDs, mapping,
thresholds and filtering policy in run provenance; resolve vans/pickups before
approving v1. Counts represent model detections visible per snapshot, not unique
vehicles across frames. A successful zero is permitted; failure is never zero.

### PROPOSED camera_window_analytics v1.0.0

Candidate location: `data/processed/camera_window_analytics_v1.parquet`.
Grain: **one camera, model run and fixed UTC window**.
Primary key: **(model_run_id, camera_id, window_start, window_minutes)**.
Window boundaries are [window_start, window_end), aligned to UTC midnight in
fixed minute increments; window_minutes must be a positive divisor of 1,440.
An agreed run's selected manifest population defines availability. Windows
without successful inference may still be emitted to expose coverage.

| Intended field | Intended Arrow type | Required semantics |
| --- | --- | --- |
| schema_version | string | Non-null, 1.0.0 |
| model_run_id | string | Non-null; do not mix model runs in one aggregate |
| camera_id | string | Non-null camera join key |
| window_start, window_end | timestamp[us, UTC] | Non-null bounds; end=start+window_minutes |
| window_minutes | int32 | Non-null positive configured window size |
| available_images | int64 | Non-null, >=0; selected manifest images with validated time in this window |
| successful_images | int64 | Non-null, >=0; successful count records in the window |
| failed_images | int64 | Non-null, >=0; unreadable or inference_error records |
| skipped_images | int64 | Non-null, >=0; explicitly skipped image records |
| unprocessed_images | int64 | Non-null, >=0; available images without a run result |
| mean_car_count | float64 | Mean successful per-snapshot car_count, null if successful_images=0 |
| mean_motorcycle_count | float64 | Mean successful per-snapshot motorcycle_count, same null rule |
| mean_bus_count | float64 | Mean successful per-snapshot bus_count, same null rule |
| mean_truck_count | float64 | Mean successful per-snapshot truck_count, same null rule |
| mean_total_vehicle_count | float64 | Mean successful total_vehicle_count, same null rule |
| successful_fraction | float64 | successful_images/available_images; null if available_images=0 |

Require available_images = successful_images + failed_images + skipped_images
+ unprocessed_images. Unknown-time rows are excluded from windows and must be
reported separately at run level. A run-level count table should expose how many
rows could not be assigned to windows; no such table currently exists.

A missing snapshot is absent availability, not a zero-vehicle observation.
Expected-image coverage cannot be inferred from a nominal 10-minute interval
without an approved operating schedule. No expected_images, traffic-volume,
speed or congestion field is proposed without a separate defensible definition.

### Approval and integration gates

Before implementation, agree on timezone evidence, class boundaries/model mapping,
thresholds, run selection, failure/retry policy and window size. Validate keys,
foreign keys, nonnegative counts, total-count equality, UTC types, coverage
invariants and version compatibility. Consumers must reject unsupported major
versions rather than reinterpret fields. External-source schemas and spatial/
temporal join rules remain undefined pending source selection and provenance.

## EXISTING EXPERIMENTAL: YOLO feasibility artifacts (Step 4)

Contract identifier: **feasibility-1.0.0**. Implemented only for bounded experiments
under `data/interim/yolo_feasibility/`. This is separate from the **PROPOSED**
production v1.0.0 schemas above: it does not implement or approve those contracts,
change the source manifest, or define dashboard/analysis interfaces.

Producer: `scripts/test_yolo_feasibility.py`, backed by
`engineering/yolo_feasibility.py` and `engineering/vehicle_detector.py`.
Each output folder holds one run and refuses overwriting nonempty folders.
See [Step 4 usage and verified run](step4_yolo_feasibility.md).

### sample_manifest.parquet

Grain: one selected source image, including skipped/failed/zero-detection images.
Key: (model_run_id, image_id); image_path is also unique within a run.
All original manifest columns are retained, with their meanings unchanged.
In particular, image_count remains the repeated expected camera image count,
not the new predicted vehicle count. Additional fields are:

| Field | Logical / Arrow type | Meaning |
| --- | --- | --- |
| schema_version | string / large_string | Constant feasibility-1.0.0 |
| model_run_id | string / large_string | Random unique run ID; not part of deterministic sample selection |
| image_id | string / large_string | SHA-256 of UTF-8 relative image_path; path identity, not content hash |
| sample_date | string / large_string | YYYY-MM-DD from naive timestamp, or unknown |
| time_band | string / large_string | Inclusive clock-hour bands 00-10, 11-15, 16-23; unknown if invalid time |
| status | string / large_string | success, skipped, missing_file, invalid_path, unreadable, inference_error |
| error_message | nullable string / large_string | Null for success, explanation otherwise; model-load errors apply to pending images |
| annotation_error | nullable string / large_string | Separate annotation-write error; inference predictions remain usable |
| annotated_image_path | nullable string / large_string | Project-relative annotated JPEG copy, including successful zero-detection images |
| width, height | nullable int64 | Original decoded pixel dimensions; null if not decoded |
| car_count, motorcycle_count, bus_count, truck_count | nullable int64 | Nonnegative predicted COCO class counts for success, null otherwise |
| total_vehicle_count | nullable int64 | Sum of four counts on success; null otherwise |
| runtime_seconds | nullable double | Per-image decode/predict/annotation elapsed wall time; null if skipped before decode |
| inference_seconds | nullable double | Predict-call elapsed time, including failed calls; null if never attempted |

timestamp stays timezone-naive timestamp[us], unchanged from the source manifest.
A zero count is a successful image with no retained predictions, never a failed
read or inference. Sampling-only images are skipped, not zero-detection successes.
No label remapping of vans/pickups is attempted; this experiment uses the model's
COCO categories. These classes remain predictions, not ground truth.

### detections.parquet

Grain: one retained vehicle prediction. Key:
(model_run_id, image_id, detection_index). Foreign key (model_run_id, image_id)
references a successful sample row; path, camera ID and timestamp must also agree.
An empty detection table still has the same typed schema.

| Field | Arrow type | Meaning |
| --- | --- | --- |
| schema_version, model_run_id | large_string | Same version/run as the parent image |
| image_id, image_path, camera_id | large_string | Parent image identity and string camera ID |
| timestamp | timestamp[us], nullable, no timezone | Parent filename clock value; null if source time invalid |
| detection_index | int64 | Zero-based prediction ordinal within image/run |
| class_id | int64 | COCO ID: 2 car, 3 motorcycle, 5 bus, 7 truck |
| vehicle_class | large_string | Corresponding COCO name |
| confidence | double | Model score in [0,1], at least configured threshold; not measured accuracy |
| x_min, y_min, x_max, y_max | double | Original decoded-image pixel xyxy bounds, upper-left origin, x right/y down |

Boxes must be finite, positive-area and inside [0,width] x [0,height].
Coordinates refer to the unrotated decoded source pixels, not the network's
resized input. No tracking IDs or unique-vehicle-throughput claims are made.

### summary.json and annotated_images/

summary.json records version/run/time, source-manifest checksum, model identifier,
local weight path and SHA-256, installed package versions, all exposed inference
parameters, population/eligible-stratum counts, sampled camera/date/time-band
coverage, status counts, successful/zero/failed image counts, annotation counts
and errors, per-class detections, and per-image error details.

Runtime units are seconds. Separate model-load/download time, per-image total
runtime and predict-call describe statistics are recorded; first-call warmup is
included. elapsed_before_output_write_seconds excludes final artifact writes.
Undefined runtime statistics are JSON null. inference_executed is false if no
predict call occurred. evaluation_metrics is null: no labelled evaluation exists.
Status is sampling_only, complete, partial_failure or failed. CLI returns nonzero
for partial_failure/failed while preserving available failure records.

annotated_images/<image_id>.jpg contains JPEG copies for successful images,
including successful zeros; raw images are never overwritten. Annotation failures
are reported separately. No image contents are uploaded to model services.

## Step 4.5A diagnostic tooling (not yet executed)

The standalone script defines separate diagnostics-0.1.0 CSV/provenance outputs
under a new data/interim/yolo_diagnostics/<run>/ directory. No diagnostic run
is claimed yet; the exact image list awaits user confirmation. See
[diagnostic inputs, configurations and output columns](step45a_yolo_diagnostics.md).
Existing feasibility and proposed production contracts are unchanged.


## Step 4.5B manual scope preparation (no metrics)

The local evaluation_scope.csv under data/interim/yolo_ground_truth has one row
per copied image, keyed by image_filename. Its exact columns are image_filename,
scope, x_min, y_min, x_max, y_max, reviewed. Scope is pending/full/roi;
reviewed is a true/false CSV value. Pending rows have blank bounds and false.
Full bounds are 0,0,1,1; ROI bounds are a manually selected positive-area
rectangle normalized to [0,1]. All eight rows initially remain pending.
No ROIs are inferred from predictions or annotation density.

The filename-only review_contact_sheet.png and review_notes.md support manual
coverage decisions. The read-only scripts/validate_evaluation_scope.py command
checks draft structure; --require-reviewed blocks evaluation until every row
is reviewed full/roi. It does not certify annotation completeness or define
ROI-border matching/clipping rules. No evaluation metrics have been produced.
The current classes.txt has car, motorcycle, bus, truck in that order; it was
already correct during scope preparation and was not edited.

## EXISTING EXPERIMENTAL: Step 4.5B ROI evaluation

The earlier preparation-only status is superseded by an executed eight-image
evaluation (November 16, 2026). New roi-evaluation-1.0.0 outputs are under
data/interim/yolo_ground_truth/evaluation/. Source labels/scopes/predictions
remain unchanged. See [evaluation rules and output schemas](step45b_roi_evaluation.md).

Membership uses inclusive normalized box centers for both GT and predictions;
boxes are not clipped before class-aware greedy matching at IoU >=0.5.
COCO 2,3,5,7 maps explicitly to GT 0,1,2,3. Duplicate unmatched predictions are FP.
CSV metrics have blank undefined ratios; JSON uses null. Count error is
prediction minus GT and count MAE averages absolute errors across eight images.
matches.csv preserves GT box indices and original prediction CSV row references.
The mixed full/ROI scopes and hand-picked images limit generalization; no
production dataset contract, inference pipeline or timezone convention changes.

## EXISTING ENGINEERING INTERMEDIATE: Step 5 raw detection

The Step 5 pipeline writes **detection-raw-1.0.0** checkpoints under a new
data/interim run directory. This is not the PROPOSED UTC-aware production contract
and does not start Step 6 cleaning or Step 7 integration.
See [Step 5 commands, schemas and recovery](step5_detection_pipeline.md).

Each committed batch directory contains image_counts.parquet (one attempted
image) and detections.parquet (one prediction). Shared identity is run_id,
image_id, image_path, string camera_id and naive timestamp[us]. Image keys are
(run_id,image_id); detection keys add a zero-based detection_index.
Image counts are nonnegative integers for success (zero is valid) and null on
failure. status/error_message expose missing, invalid, unreadable and inference
errors. Source bytes have a separate source_sha256; processed_at is UTC while
capture timestamp is unchanged. Detection class IDs remain COCO 2,3,5,7, with
confidence and source-oriented pixel xyxy boxes.

Parquet files have explicit fixed schemas even for empty tables. Batch commits
publish image and detection tables together; resume uses checksums and strict
manifest/weights/configuration/software/source fingerprints to prevent mixing
runs or duplicate publication. This only establishes pipeline integrity, not
semantic cleaning or detection accuracy.

### Step 5 full-run artifact location

The completed baseline run is under
data/interim/detection_runs/yolov8n_baseline_full_01/: 61,319 successful image rows,
274,096 detection rows and 614 committed paired-Parquet checkpoints.
Existing detection-raw-1.0.0 schemas are unchanged. The additional
failed_images.parquet uses the same image schema and is empty (zero failures);
execution_report.json records preflight, command, elapsed time and operational
integrity checks. Neither is a Step 6 cleaned output or Step 7 integration
contract. See the completed-run section of [Step 5 documentation](step5_detection_pipeline.md).


## EXISTING: Step 6 detection validation artifacts

The executed validator writes **detection-validation-1.0.0** reports under
`data/interim/detection_validation/`; it does not alter detection-raw-1.0.0 or
implement the PROPOSED Step 7 contracts. See [checks, CLI and verified results](step6_detection_validation.md).

`data_quality_report.json` records source provenance, validator hash, coverage,
row/class/zero counts, null counts by table, individual checks, critical/warning
issue counts by category, elapsed validation seconds and readiness recommendation.
`data_quality_report.md` is the concise human report.
`validation_issues.parquet` has one row per rule violation on a source row or
one dataset/checkpoint-level finding. Its fields are severity/category/table/message
(string), row_index/detection_index (nullable int64), image_id/image_path
(nullable string). Row indices refer to the loaded manifest or ordered
concatenated checkpoint tables; global findings have null row references.
Repeated findings can refer to one image; issue counts are not unique image counts.
No cleaned predictions are produced or silently substituted.

Verified: 61,319 image rows, 274,096 prediction rows, 614 checkpoints, no missing
or duplicate keys, 10,488 successful zeros, zero processing failures, zero critical
issues and three warnings (optional English metadata, model limitations and
unverified timestamp provenance). Structurally ready for Step 7 planning;
production contract/timezone approval and integration remain separate.


## IMPLEMENTED: Step 7 consumer contract ? camera-observations-1.0.0

This contract is the current analysis/dashboard handoff, exported on November 17,
2026. It preserves naive capture timestamps as explicitly requested and does
**not** implement the earlier PROPOSED UTC-aware/window schemas. The latter
remain designs requiring evidence and consumer agreement. Producer:
`engineering/export_datasets.py`, CLI `scripts/export_datasets.py`.
Exact semantics, publication/recovery rules, provenance and working join examples
are in [Step 7 handoff](step7_data_handoff.md).

Version is stored as Arrow `contract_version` metadata in each Parquet file and
in `data/processed/export_manifest.json`, not copied from source schema_version.
Source detection-raw-1.0.0 remains recorded in export run provenance. All output
fields below are non-null unless marked nullable. All camera IDs are unchanged
strings; capture timestamp values and microsecond precision are unchanged.

### data/processed/camera_snapshots.parquet

Grain: one successful image/run, including zero detections. Key and ordering:
**(run_id,image_id)**. Current rows/unique keys: **61,319 / 61,319**.

| Fields | Arrow type | Nullable |
| --- | --- | --- |
| run_id, image_id, image_path, camera_id | string | no |
| timestamp | timestamp[us], no timezone | yes |
| source_sha256 | string | no |
| width, height | int64 | no |
| car_count, motorcycle_count, bus_count, truck_count, total_vehicle_count | int64 | no |
| status | string | no |
| processed_at | timestamp[us, UTC] | no |

image_id is SHA-256 of relative image_path, not content. source_sha256 is the
recorded image-byte checksum. Dimensions refer to the Step 5 EXIF-normalized
decode. status is always success; counts are nonnegative model predictions,
and total equals the sum of four class counts. There are **10,488 valid zeros**.
Failures stay in Step 5 with null counts; the export manifest records excluded
failure count (currently zero). Capture timestamp is an uncertain filename clock
reading; processed_at is the separate known processing instant.

### data/processed/vehicle_detections.parquet

Grain: one original prediction box. Key and ordering:
**(run_id,image_id,detection_index)**. Current rows/unique keys:
**274,096 / 274,096**. FK (run_id,image_id) references camera_snapshots.

| Fields | Arrow type | Nullable |
| --- | --- | --- |
| run_id, image_id, image_path, camera_id | string | no |
| timestamp | timestamp[us], no timezone | yes |
| detection_index, class_id | int64 | no |
| vehicle_class | string | no |
| confidence, x_min, y_min, x_max, y_max | float64 | no |

All values are preserved. detection_index is the zero-based source ordinal.
COCO mapping is 2=car, 3=motorcycle, 5=bus, 7=truck. Confidence is a model score,
not accuracy. Boxes are finite positive-area pixel xyxy within parent dimensions,
upper-left origin, x right/y down; not normalized or resized network coordinates.
Timestamp/path/camera must equal the parent, including null timestamp agreement.
No prediction removal, clipping, rounding, class remapping or duplicate-box
suppression is performed. Location metadata is not repeated on detection rows.

### data/processed/camera_metadata.parquet

Grain/key/order: one row per **camera_id** (100 rows and unique keys). Snapshot
camera_id is an FK to this table. CSV and manifest metadata are checked for
agreement; conflicts block export. Fields preserve all nine CSV source columns.

| Fields | Arrow type | Nullable |
| --- | --- | --- |
| camera_id, location | string | no |
| location_detail, location_en, location_detail_en, ip_address | string | yes |
| latitude, longitude | float64 | no |
| image_count | int64 | no |

English nulls remain one location_en and two location_detail_en; no fabricated
translations. Coordinates are source decimal degrees with unverified CRS/accuracy.
ip_address is metadata, not a geographic join key. image_count is expected
available images per camera, never vehicle counts; do not sum after repeating it
in a snapshot join. Camera metadata is normalized here rather than repeated per box.

### Export integrity and consumer semantics

Snapshot-to-camera joins use camera_id (many-to-one); detection-to-snapshot joins
use BOTH run_id and image_id (many-to-one). Do not key joins by timestamps.
Use snapshots as the availability/count population so zero detections survive.
`export_manifest.json`, published last, records completion, source run/baseline,
weights/class mapping, Step 6 report status/hash/warnings, source and output
fingerprints, software, sizes, rows and unique keys. `read_exports` verifies the
completion marker, schemas and checksums before returning consumer tables.
Identical reruns do not rewrite files. Incompatible existing files are refused;
interrupted matching partial exports can resume without overwriting data.

Structural and serialization checks passed on the full dataset, with zero
broken keys/references/count inconsistencies. The original three Step 6 warnings
remain. This handoff does not validate detection accuracy. Counts estimate visible
vehicles per snapshot, not traffic flow, unique throughput, speed or congestion.
Timezone/clock evidence and any future UTC/window contract remain unresolved;
no silent timezone conversion is permitted. Consumer-specific analysis and UI
are outside this export implementation.
