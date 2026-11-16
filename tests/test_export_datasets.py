"""Synthetic Step 7 integration tests: no pixels, model or downloads."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bangkok_cctv.engineering import export_datasets as exports
from bangkok_cctv.engineering.detection_checkpoints import commit_batch, sha256
from bangkok_cctv.engineering.detection_validation import validate_run
from bangkok_cctv.engineering.pipeline import (
    BASELINE, COUNTS, DETECTION_SCHEMA, IMAGE_SCHEMA, VERSION,
)


@pytest.fixture
def inputs(tmp_path):
    interim = tmp_path / "data/interim"
    interim.mkdir(parents=True)
    cameras = pd.DataFrame({"camera_id": ["001", "002"], "location": ["A", "B"],
                           "location_detail": ["detail A", "detail B"],
                           "location_en": [None, "B English"],
                           "location_detail_en": [None, "detail B English"],
                           "latitude": [13.7, 13.8], "longitude": [100.5, 100.6],
                           "ip_address": ["10.0.0.1", "10.0.0.2"], "image_count": [1, 1]})
    manifest = cameras.copy()
    manifest["image_path"] = ["data/raw/001/a.jpg", "data/raw/002/b.jpg"]
    manifest["timestamp"] = pd.Series([pd.Timestamp("2026-08-25 13:15:54")] * 2,
                                       dtype="datetime64[us]")
    manifest["metadata_matched"] = True
    manifest_path = interim / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    camera_path = tmp_path / "data/raw/cctv_info.csv"
    camera_path.parent.mkdir(parents=True)
    cameras.to_csv(camera_path, index=False)
    rows = []
    for index, record in enumerate(manifest[["image_path", "camera_id", "timestamp"]].to_dict("records")):
        rows.append({"schema_version": VERSION, "run_id": "test-run", **record,
                     "image_id": hashlib.sha256(record["image_path"].encode()).hexdigest(),
                     "source_index": index, "status": "success", "error_message": None,
                     "source_sha256": "a" * 64, "width": 100, "height": 80,
                     **{c: int(index == 0 and c in ("car_count", "total_vehicle_count")) for c in COUNTS},
                     "runtime_seconds": .2, "inference_seconds": .1,
                     "processed_at": pd.Timestamp("2026-11-16", tz="UTC")})
    image_table = pa.Table.from_pylist(rows, schema=IMAGE_SCHEMA)
    identity = {c: rows[0][c] for c in DETECTION_SCHEMA.names[:6]}
    detection_table = pa.Table.from_pylist([
        {**identity, "detection_index": 0, "class_id": 2, "vehicle_class": "car",
         "confidence": .8, "x_min": 1., "y_min": 2., "x_max": 30., "y_max": 40.}
    ], schema=DETECTION_SCHEMA)
    run = interim / "run"
    (run / "batches").mkdir(parents=True)
    commit_batch(run / "batches", 0, image_table, detection_table, "test-run", .5)
    metadata = {"schema_version": VERSION, "run_id": "test-run", "target_images": 2,
                "signature": {"schema_version": VERSION, "baseline": BASELINE,
                              "manifest_sha256": sha256(manifest_path), "weights_sha256": "b" * 64,
                              "image_limit": None, "batch_size": 100}}
    summary = {"schema_version": VERSION, "run_id": "test-run", "status": "complete",
               "target_images": 2, "processed_images": 2, "remaining_images": 0,
               "committed_batches": 1, "detection_count": 1,
               "successful_zero_detection_images": 1, "status_counts": {"success": 2},
               "per_class_detection_counts": {"car": 1, "motorcycle": 0, "bus": 0, "truck": 0}}
    (run / "run.json").write_text(json.dumps(metadata))
    (run / "summary.json").write_text(json.dumps(summary))
    validation = interim / "validation"
    validate_run(tmp_path, run, manifest_path, validation)
    options = {"project_root": tmp_path, "run_dir": run, "manifest_path": manifest_path,
               "camera_csv": camera_path, "validation_report": validation / "data_quality_report.json",
               "output_dir": tmp_path / "data/processed"}
    return options, manifest, image_table.to_pandas(), detection_table.to_pandas(), cameras


def test_full_export_preserves_values_nulls_zeros_and_inputs(inputs):
    options, manifest, images, detections, cameras = inputs
    protected = {p: sha256(p) for p in options["project_root"].rglob("*") if p.is_file()}
    result = exports.export_datasets(**options)
    tables = exports.read_exports(options["output_dir"])
    exports.validate_exports(tables, images, detections)
    assert result["successful_zero_detection_images"] == 1
    assert tables["camera_snapshots"]["timestamp"].type == pa.timestamp("us")
    assert set(tables["camera_snapshots"]["camera_id"].to_pylist()) == {"001", "002"}
    assert tables["camera_metadata"]["location_en"].null_count == 1
    assert tables["camera_metadata"]["location_detail_en"].null_count == 1
    assert {p: sha256(p) for p in protected} == protected


def test_rerun_byte_identical_no_rewrite(inputs):
    options = inputs[0]
    first = exports.export_datasets(**options)
    out = options["output_dir"]
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in out.iterdir() if p.name != "run.lock"}
    second = exports.export_datasets(**options)
    assert second == first
    assert {p.name: (sha256(p), p.stat().st_mtime_ns) for p in out.iterdir() if p.name != "run.lock"} == before


def test_interrupted_publication_is_resumable(inputs, monkeypatch):
    options = inputs[0]
    real = exports._publish_file
    calls = 0
    def interrupted(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publication interruption")
        real(source, target)
    monkeypatch.setattr(exports, "_publish_file", interrupted)
    with pytest.raises(OSError, match="interruption"):
        exports.export_datasets(**options)
    assert not (options["output_dir"] / "export_manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        exports.read_exports(options["output_dir"])
    published = options["output_dir"] / "camera_snapshots.parquet"
    before = sha256(published), published.stat().st_mtime_ns
    monkeypatch.setattr(exports, "_publish_file", real)
    exports.export_datasets(**options)
    assert (sha256(published), published.stat().st_mtime_ns) == before
    assert len(exports.read_exports(options["output_dir"])) == 3


def test_incompatible_existing_output_never_overwritten(inputs):
    options = inputs[0]
    out = options["output_dir"]
    out.mkdir()
    incompatible = out / "vehicle_detections.parquet"
    incompatible.write_bytes(b"existing work")
    with pytest.raises(ValueError, match="incompatible"):
        exports.export_datasets(**options)
    assert incompatible.read_bytes() == b"existing work"
    assert not (out / "camera_snapshots.parquet").exists()


@pytest.mark.parametrize("problem", ["failed", "stale_manifest", "stale_summary", "wrong_run", "corrupt_batch"])
def test_validation_gate(inputs, problem):
    options = inputs[0]
    if problem in ("failed", "wrong_run"):
        path = options["validation_report"]
        report = json.loads(path.read_text())
        if problem == "failed":
            report["issue_counts"]["critical"] = 1
        else:
            report["provenance"]["run_id"] = "other-run"
        path.write_text(json.dumps(report))
    elif problem == "stale_manifest":
        with options["manifest_path"].open("ab") as stream:
            stream.write(b"changed")
    elif problem == "stale_summary":
        (options["run_dir"] / "summary.json").write_text("{}")
    else:
        batch = next((options["run_dir"] / "batches").glob("batch-*"))
        with (batch / "detections.parquet").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        exports.export_datasets(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("problem", ["duplicate_camera", "conflicting_location", "missing_camera"])
def test_metadata_conflicts(inputs, problem):
    options, m, i, d, c = inputs
    if problem == "duplicate_camera":
        c = pd.concat([c, c.iloc[:1]], ignore_index=True)
    elif problem == "conflicting_location":
        c.loc[0, "location"] = "different"
    else:
        c = c.iloc[:1]
    with pytest.raises(ValueError):
        exports.build_exports(m, i, d, c, "test-run")


def test_reordered_inputs_produce_same_exports(inputs):
    _, m, i, d, c = inputs
    original = exports.build_exports(m, i, d, c, "test-run")
    shuffled = exports.build_exports(m.iloc[::-1], i.iloc[::-1], d.iloc[::-1], c.iloc[::-1], "test-run")
    assert all(original[name].equals(shuffled[name], check_metadata=True) for name in original)


def test_modified_export_rejected(inputs):
    options = inputs[0]
    exports.export_datasets(**options)
    path = options["output_dir"] / "camera_snapshots.parquet"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        exports.read_exports(options["output_dir"])


@pytest.mark.parametrize("change", ["count", "bbox", "class", "timestamp", "duplicate", "schema", "null"])
def test_export_validation_rejects_altered_values(inputs, change):
    _, m, i, d, c = inputs
    tables = exports.build_exports(m, i, d, c, "test-run")
    name = "camera_snapshots" if change in {"count", "timestamp", "null"} else "vehicle_detections"
    frame = tables[name].to_pandas()
    if change == "count":
        frame.loc[0, "total_vehicle_count"] += 1
    elif change == "timestamp":
        frame["timestamp"] += pd.Timedelta(hours=7)
    elif change == "bbox":
        frame.loc[0, "x_max"] += 1
    elif change == "class":
        frame.loc[0, "class_id"] = 7
    elif change == "null":
        frame.loc[0, "camera_id"] = None
    elif change == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    if change == "null":
        index = tables[name].schema.get_field_index("camera_id")
        tables[name] = tables[name].set_column(
            index, tables[name].schema.field(index), pa.array(frame.camera_id, type=pa.string()))
    elif change == "schema":
        tables[name] = tables[name].drop(["confidence"])
    else:
        tables[name] = pa.Table.from_pandas(frame, schema=exports.SCHEMAS[name], preserve_index=False).replace_schema_metadata(exports.SCHEMAS[name].metadata)
    with pytest.raises(ValueError):
        exports.validate_exports(tables, i, d)

