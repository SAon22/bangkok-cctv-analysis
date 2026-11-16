# Step 7: consumer data contract and handoff

Contract: **camera-observations-1.0.0**. This implements the requested unchanged,
naive capture-time export. It is separate from the earlier **PROPOSED** UTC-aware
image_vehicle_counts/window contracts, which remain unimplemented. No source
schema, predictions, raw files or earlier pipeline behavior changes.

## Run and publication

```powershell
python -B scripts/export_datasets.py
```

Defaults use the full Step 5 run, existing manifest, raw camera CSV, passed Step 6
JSON report and `data/processed/`. Actual options are `--project-root`, `--run-dir`,
`--manifest-path`, `--camera-csv`, `--validation-report` and `--output-dir`.
Relative inputs resolve against the project root. Alternative export directories
must remain under `data/processed/`.

Export requires a passed Step 6 report (warnings allowed), matching run identity
and unchanged report-recorded manifest/run/summary hashes. Checkpoint checksums,
schemas, sequence and counts are checked again. Structural validation is reused
on the current tables. Camera CSV values must agree with all manifest metadata;
conflicts block export rather than choosing a source silently. Each checkpoint
Parquet is parsed once, plus a byte pass for its committed checksum. No model,
image decode or raw pixel access is needed.

Outputs are staged on the same filesystem, serialized, read back and validated.
An OS single-writer lock protects publication. Each file is published using an
exclusive atomic hard link, followed **last** by `export_manifest.json` as the
completion marker. This requires a local filesystem supporting hard links.
Files are never replaced. All existing targets are checked for compatibility
before publication. An identical rerun leaves output bytes and mtimes unchanged;
an interrupted partial publication can reuse matching files. Changed inputs,
exporter/library versions or incompatible files require a fresh `--output-dir`.
Readers must require the completion marker; use `read_exports` below to check
schemas and checksums. This is not a filesystem-wide atomic multi-file swap,
a backup or a guarantee against every power-loss/filesystem failure.
`run.lock` is an inert lock file when no export is running; it is not a dataset.

## Exact contract

All files carry Arrow metadata `contract_version=camera-observations-1.0.0`,
`table=<table_name>` and an explicit capture-time uncertainty policy. The JSON
completion manifest records the same version. Keys never use timestamps.
Breaking changes to grain, keys, class or time meaning require a new major
contract; additive nullable fields require a documented minor version.

### camera_snapshots.parquet

Grain: **one successful image per model run**, including zero-prediction images.
Primary key: **(run_id, image_id)**. Sorted by that key.

| Fields | Arrow type | Nullable | Semantics |
| --- | --- | --- | --- |
| run_id | string | no | Unchanged Step 5 run ID; provenance in export_manifest.json |
| image_id | string | no | Unchanged SHA-256 of UTF-8 relative image_path; path identity, not content |
| image_path | string | no | Unchanged project-relative POSIX path to original image |
| camera_id | string | no | Unchanged source ID; foreign key to camera_metadata; preserve leading zeros |
| timestamp | timestamp[us], no timezone | yes | Unchanged filename clock reading; null only if source parsing was invalid |
| source_sha256 | string | no | Recorded SHA-256 of original image bytes; not rehashed by export |
| width, height | int64 | no | Positive Step 5 EXIF-normalized decoded-image dimensions in pixels |
| car_count, motorcycle_count, bus_count, truck_count | int64 | no | Nonnegative retained predictions by COCO vehicle class |
| total_vehicle_count | int64 | no | Sum of four class counts; zero is a valid successful observation |
| status | string | no | Always success; this table intentionally exports only successful images |
| processed_at | timestamp[us, UTC] | no | Unchanged inference processing instant, distinct from uncertain capture time |

Failed images never become zeros. Their rows remain in immutable Step 5 inputs;
the completion manifest records excluded_failed_images. The current full run
has zero failures. Runtime, source ordering and error fields remain available in
Step 5; they are operational metadata, not required in the consumer table.

### vehicle_detections.parquet

Grain: **one retained YOLO box**. Primary key:
**(run_id, image_id, detection_index)**. Sorted by that key.

| Fields | Arrow type | Nullable | Semantics |
| --- | --- | --- | --- |
| run_id, image_id | string | no | Composite foreign key to camera_snapshots |
| image_path, camera_id | string | no | Exact parent/source values for traceability; no repeated location metadata |
| timestamp | timestamp[us], no timezone | yes | Exact parent/source clock reading, including null agreement |
| detection_index | int64 | no | Original zero-based ordinal within the image/run, not a tracking ID |
| class_id | int64 | no | COCO 2=car, 3=motorcycle, 5=bus, 7=truck |
| vehicle_class | string | no | Exact corresponding model class name; no reinterpretation of vans/pickups |
| confidence | float64 | no | Unchanged finite model score in [0,1], not measured accuracy |
| x_min, y_min, x_max, y_max | float64 | no | Unchanged finite positive-area pixel xyxy boxes in recorded decoded-image coordinates |

Origin is upper-left, x increases right, y down. Bounds are inside [0,width]
and [0,height] of the referenced snapshot. Boxes use the Step 5 EXIF-normalized
decode orientation; they are not network-resized or normalized [0,1] coordinates.
Overlapping predictions are retained. No clipping, rounding, threshold changes,
deduplication, tracking, class remapping or confidence recalibration is applied.

### camera_metadata.parquet

Grain/key: **one row per camera_id**; sorted by string camera_id. This table is
justified by 100 shared camera metadata records referenced by 61,319 snapshots.
It preserves all nine CSV columns once, avoiding location/coordinate repetition
on detection rows. Values must agree with the manifest's metadata snapshot.

| Fields | Arrow type | Nullable | Semantics |
| --- | --- | --- | --- |
| camera_id | string | no | Source join key, never numeric |
| location | string | no | Source location label, not independently verified |
| location_detail | string | yes | Original detailed location description |
| location_en, location_detail_en | string | yes | Original optional English labels; never imputed |
| latitude, longitude | float64 | no | Finite source decimal degrees; CRS/geolocation accuracy unverified |
| ip_address | string | yes | Original source metadata, not a geographic join key |
| image_count | int64 | no | Source expected available image count per camera; not vehicle predictions |

Do not sum metadata image_count after joining it onto every snapshot. Join
snapshots to camera metadata on camera_id with many-to-one validation; join
detections to snapshots on BOTH run_id and image_id with many-to-one validation.
Do not join solely on camera_id/timestamp. More than one image may legitimately
share a clock reading. Count snapshots from the snapshot table so successful
zeros are not lost through an inner join to detections.

## Provenance and validation

`export_manifest.json` contains the source run metadata/signature, baseline,
weights hash, library versions, COCO mapping, Step 6 report hash/status/warnings,
input and checkpoint-marker fingerprints, exporter hash, exact output sizes,
checksums, row/unique-key counts, excluded failures and preserved zero count.
No changing export wall-clock value is embedded, ensuring identical reruns.

Fresh checks validate snapshot/detection keys, references, per-class and total
counts, zero preservation, exact field values/dtypes/nullability, camera metadata
agreement and serialized Parquet roundtrips. Snapshots/detections are exact
projections of successful Step 5 rows, with only sorting and explicit schema
metadata. Input schema_version remains recorded in run provenance; output
contract_version is Arrow/JSON metadata rather than a misleading copied row value.

## Minimal consumer examples

Run from the repository root (e.g. a notebook one directory below the root can
resolve it with `Path.cwd().parent` after checking pyproject.toml exists):

```python
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
assert (ROOT / "pyproject.toml").is_file(), "Set ROOT to the repository root"
sys.path.insert(0, str(ROOT / "src"))
from bangkok_cctv.engineering.export_datasets import read_exports

tables = read_exports(ROOT / "data/processed")  # completion/schema/hash gate
snapshots = tables["camera_snapshots"].to_pandas()
detections = tables["vehicle_detections"].to_pandas()
cameras = tables["camera_metadata"].to_pandas()

# Data Science: preserve every successful snapshot, including zeros.
snapshot_locations = snapshots.merge(cameras, on="camera_id", validate="many_to_one")

# Dashboard: retrieve source dimensions and locations for individual predictions.
objects = detections.merge(
    snapshots[["run_id", "image_id", "width", "height"]],
    on=["run_id", "image_id"], validate="many_to_one",
).merge(cameras, on="camera_id", validate="many_to_one")
assert len(snapshot_locations) == len(snapshots)
```

After the completion gate, direct `pandas.read_parquet(ROOT / "data/processed/...parquet")`
is also supported. This document supplies examples; it does not implement Data
Science analysis or Dashboard features.

## Unresolved evidence and limitations

Capture timezone and camera-clock accuracy remain unverified. These exports
preserve naive timestamps as explicitly requested. A future confirmed timezone
may support a separate UTC conversion contract; simply assuming Asia/Bangkok or
UTC would change semantics and requires agreement. No inferred UTC column or
invented source timezone is added. English nulls, camera coordinate uncertainty,
missed/incorrect/overlapping pretrained predictions and the hand-picked nature
of the earlier eight-image evaluation remain limitations.
Counts estimate vehicles visible in snapshots. They do not measure traffic flow,
unique throughput, speed, or validated congestion. Consumer acceptance of these
limitations and any future time-window analytics definition require team review.


## Verified completion (November 17, 2026)

Executed `python -B scripts/export_datasets.py` against source run
`15c14accabe04a87a03b02073ae332e3`; exit code 0. Step 6 provenance remained
passed_with_warnings (zero critical issues, three warnings). Exact artifacts:

| File under data/processed/ | Rows | Unique primary keys | Bytes |
| --- | ---: | ---: | ---: |
| camera_snapshots.parquet | 61,319 | 61,319 | 10,862,578 |
| vehicle_detections.parquet | 274,096 | 274,096 | 19,001,830 |
| camera_metadata.parquet | 100 | 100 | 15,240 |
| export_manifest.json | provenance | n/a | 4,344 |

All 10,488 successful zero-detection snapshots were retained. No failed images
were excluded (the source had none). Keys and references are valid; per-image
and per-class counts match the unmodified source predictions. All original
classes, confidences, boxes, timestamps and camera IDs were preserved. Counts by
class remain car 208,919; motorcycle 12,592; bus 29,161; truck 23,424.

Ran the export command again: exit code 0, all four published files remained
byte-identical with unchanged modification times. The consumer reader and both
documented many-to-one joins were executed successfully against the full output.
Naive timestamps remained datetime64[us]. Camera English null counts remain
one location_en and two location_detail_en; no imputation was performed.
A before/after SHA-256 inventory of 1,995 protected Step 5, Step 6, feasibility,
diagnostic, evaluation, smoke and input files was identical. Raw CCTV pixels
were never read. No prior pipeline modules or outputs were modified.

Executed `python -B -m pytest -q -p no:cacheprovider` with external pytest plugin
autoload disabled: **174 passed in 19.01 seconds**, including 21 new export cases.
The new tests exercise metadata conflicts, exact value/schema/null preservation,
zeros, key/count consistency, stale/failed validation gates, checkpoint corruption,
input ordering, input immutability, interruption recovery and identical reruns.
No dependencies, models or data were downloaded. No inference, Data Science or
Dashboard feature, commit or push was performed.
