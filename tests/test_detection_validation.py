"""Small synthetic datasets exercise validation without weights or inference."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bangkok_cctv.engineering.detection_checkpoints import commit_batch, sha256
from bangkok_cctv.engineering.detection_validation import validate_run, validate_tables
from bangkok_cctv.engineering.pipeline import (
    BASELINE, COUNTS, DETECTION_SCHEMA, IMAGE_SCHEMA, VERSION,
)


@pytest.fixture
def dataset():
    stamp = pd.Timestamp("2026-08-25 06:00:00")
    manifest = pd.DataFrame({"image_path": ["data/raw/001/a.jpg", "data/raw/002/b.jpg"],
                             "camera_id": ["001", "002"], "timestamp": [stamp, stamp]})
    image_rows = []
    for index, row in enumerate(manifest.to_dict("records")):
        image_rows.append({"schema_version": VERSION, "run_id": "run", **row,
                           "image_id": hashlib.sha256(row["image_path"].encode()).hexdigest(),
                           "source_index": index, "status": "success", "error_message": None,
                           "source_sha256": "a" * 64, "width": 100, "height": 80,
                           **{c: int(index == 0 and c in ("car_count", "total_vehicle_count")) for c in COUNTS},
                           "runtime_seconds": .2, "inference_seconds": .1,
                           "processed_at": pd.Timestamp("2026-11-16", tz="UTC")})
    images = pa.Table.from_pylist(image_rows, schema=IMAGE_SCHEMA).to_pandas()
    identity = {name: image_rows[0][name] for name in DETECTION_SCHEMA.names[:6]}
    detections = pa.Table.from_pylist([
        {**identity, "detection_index": 0, "class_id": 2, "vehicle_class": "car",
         "confidence": .8, "x_min": 10., "y_min": 10., "x_max": 30., "y_max": 40.}
    ], schema=DETECTION_SCHEMA).to_pandas()
    return manifest, images, detections


def categories(dataset):
    findings = validate_tables(*dataset, "run")
    return {c["category"] for c in findings.checks
            if c["status"] == "fail" and c["severity"] == "critical"}


def test_valid_and_successful_zero(dataset):
    assert categories(dataset) == set()
    assert len(validate_tables(*dataset, "run").table()) == 0


@pytest.mark.parametrize("problem, expected", [
    ("missing_image", "missing_manifest_images"),
    ("null_key", "images_required_values"),
    ("missing_column", "images_columns"),
    ("duplicate_image", "images_keys"),
    ("duplicate_detection", "detections_keys"),
    ("bad_class", "supported_classes"),
    ("bad_class_name", "supported_classes"),
    ("count", "count_consistency"),
    ("orphan", "detection_parent"),
    ("camera", "manifest_camera_id"),
    ("timestamp", "manifest_timestamp"),
    ("source_hash", "source_hash_format"),
    ("null_success_count", "success_counts"),
    ("failed_zero", "failure_counts"),
    ("no_error_message", "error_messages"),
    ("pending", "processing_status"),
])
def test_broken_records(dataset, problem, expected):
    m, i, d = dataset
    if problem == "missing_image":
        i = i.iloc[:1].copy()
    elif problem == "null_key":
        i.loc[0, "run_id"] = None
    elif problem == "missing_column":
        i = i.drop(columns="image_id")
    elif problem == "duplicate_image":
        i = pd.concat([i, i.iloc[:1]], ignore_index=True)
    elif problem == "duplicate_detection":
        d = pd.concat([d, d], ignore_index=True)
    elif problem == "bad_class":
        d.loc[0, "class_id"] = 0
    elif problem == "bad_class_name":
        d.loc[0, "vehicle_class"] = "truck"
    elif problem == "count":
        i.loc[0, "car_count"] = 2
    elif problem == "orphan":
        d.loc[0, "image_id"] = "b" * 64
    elif problem == "camera":
        i.loc[0, "camera_id"] = "999"
    elif problem == "timestamp":
        i.loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    elif problem == "source_hash":
        i.loc[0, "source_sha256"] = "invalid"
    elif problem == "null_success_count":
        i["car_count"] = i.car_count.astype("Int64")
        i.loc[0, "car_count"] = pd.NA
    elif problem == "failed_zero":
        i.loc[1, ["status", "error_message"]] = ["unreadable", "bad image"]
    elif problem == "no_error_message":
        i.loc[1, "status"] = "unreadable"
    elif problem == "pending":
        i.loc[1, "status"] = "pending"
    assert expected in categories((m, i, d))


@pytest.mark.parametrize("field,value", [
    ("x_min", -1.), ("x_min", 30.), ("x_max", 101.),
    ("y_max", 81.), ("y_min", float("nan")), ("x_max", float("inf")),
])
def test_invalid_boxes(dataset, field, value):
    dataset[2].loc[0, field] = value
    assert "bounding_boxes" in categories(dataset)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence(dataset, value):
    dataset[2].loc[0, "confidence"] = value
    assert "confidence" in categories(dataset)


def test_failure_nullable_counts_and_no_predictions(dataset):
    m, i, d = dataset
    i.loc[1, ["status", "error_message"]] = ["missing_file", "not found"]
    for field in COUNTS:
        i[field] = i[field].astype("Int64")
        i.loc[1, field] = pd.NA
    i.loc[1, ["source_sha256", "width", "height", "inference_seconds"]] = None
    assert categories(dataset) == set()
    issues = validate_tables(m, i, d, "run").table().to_pandas()
    assert issues.category.tolist() == ["processing_failures"]


def test_all_successful_zeros_with_empty_detection_schema(dataset):
    m, i, d = dataset
    i[COUNTS] = 0
    assert categories((m, i, d.iloc[:0])) == set()


def write_run(tmp_path: Path, dataset):
    m, i, d = dataset
    manifest = tmp_path / "data/interim/image_manifest.parquet"
    manifest.parent.mkdir(parents=True)
    m.to_parquet(manifest, index=False)
    run = tmp_path / "data/interim/run"
    batches = run / "batches"
    batches.mkdir(parents=True)
    commit_batch(batches, 0, pa.Table.from_pandas(i, schema=IMAGE_SCHEMA, preserve_index=False),
                 pa.Table.from_pandas(d, schema=DETECTION_SCHEMA, preserve_index=False), "run", .4)
    metadata = {"schema_version": VERSION, "run_id": "run", "target_images": 2,
                "signature": {"schema_version": VERSION, "manifest_sha256": sha256(manifest),
                              "baseline": BASELINE, "image_limit": None,
                              "weights_sha256": "a" * 64, "batch_size": 100}}
    summary = {"schema_version": VERSION, "run_id": "run", "status": "complete",
               "target_images": 2, "processed_images": 2, "remaining_images": 0,
               "committed_batches": 1, "detection_count": 1,
               "successful_zero_detection_images": 1, "status_counts": {"success": 2},
               "per_class_detection_counts": {"car": 1, "motorcycle": 0, "bus": 0, "truck": 0}}
    (run / "run.json").write_text(json.dumps(metadata))
    (run / "summary.json").write_text(json.dumps(summary))
    return run, manifest, tmp_path / "data/interim/validation"


def test_run_reports_and_no_overwrite(tmp_path, dataset):
    run, manifest, out = write_run(tmp_path, dataset)
    before = {p: sha256(p) for p in run.rglob("*") if p.is_file()}
    report = validate_run(tmp_path, run, manifest, out)
    assert report["issue_counts"] == {"critical": 0, "warning": 3}
    assert report["ready_for_step7_planning"]
    assert report["statistics"]["zero_detection_images"] == 1
    assert pq.read_table(out / "validation_issues.parquet").num_rows == 3
    assert (out / "data_quality_report.md").is_file()
    assert {p: sha256(p) for p in before} == before
    with pytest.raises(FileExistsError):
        validate_run(tmp_path, run, manifest, out)


@pytest.mark.parametrize("problem,expected", [
    ("corrupt", "checkpoint_checksum"), ("schema", "checkpoint_schema"),
    ("summary", "final_summary"), ("pending", "finalized_checkpoints"),
    ("manifest", "manifest_fingerprint"),
])
def test_run_integrity_errors(tmp_path, dataset, problem, expected):
    run, manifest, out = write_run(tmp_path, dataset)
    batch = next((run / "batches").glob("batch-*"))
    if problem == "corrupt":
        marker_path = batch / "commit.json"
        marker = json.loads(marker_path.read_text())
        marker["image_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker))
    elif problem == "schema":
        pq.write_table(pa.table({"wrong": [1]}), batch / "detections.parquet")
    elif problem == "summary":
        (run / "summary.json").write_text("{}")
    elif problem == "pending":
        (run / "batches/.pending-interrupted").mkdir()
    elif problem == "manifest":
        m = pd.read_parquet(manifest)
        m.loc[0, "camera_id"] = "999"
        m.to_parquet(manifest, index=False)
    report = validate_run(tmp_path, run, manifest, out)
    assert report["issue_counts"]["critical"] > 0
    assert expected in report["issues_by_category"]["critical"]
    assert not report["ready_for_step7_planning"]



def test_nullable_timestamp_is_preserved(dataset):
    m, i, d = dataset
    m["timestamp"] = pd.NaT
    i["timestamp"] = pd.NaT
    d["timestamp"] = pd.NaT
    assert categories(dataset) == set()


def test_detection_null_key(dataset):
    dataset[2].loc[0, "run_id"] = None
    assert "detections_required_values" in categories(dataset)


def test_detection_metadata_mismatch(dataset):
    dataset[2].loc[0, "camera_id"] = "999"
    assert "detection_camera_id" in categories(dataset)


def test_source_hash_conflict_not_silently_deduplicated(dataset):
    m, i, d = dataset
    duplicate = i.iloc[:1].copy()
    duplicate["source_sha256"] = "b" * 64
    i = pd.concat([i, duplicate], ignore_index=True)
    assert "source_hash_consistency" in categories((m, i, d))
