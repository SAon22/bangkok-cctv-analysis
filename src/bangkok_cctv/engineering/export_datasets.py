"""Step 7: deterministic, validated observation exports; no inference or pixels."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .detection_checkpoints import sha256, writer_lock
from .detection_validation import KEY, validate_tables
from .pipeline import COUNTS, DETECTION_SCHEMA, IMAGE_SCHEMA

LOG = logging.getLogger(__name__)
CONTRACT = "camera-observations-1.0.0"
CAMERA_FIELDS = ["camera_id", "location", "location_detail", "location_en",
                 "location_detail_en", "latitude", "longitude", "ip_address", "image_count"]
SNAPSHOT_FIELDS = ["run_id", "image_id", "image_path", "camera_id", "timestamp",
                   "source_sha256", "width", "height", *COUNTS, "status", "processed_at"]
DETECTION_FIELDS = [name for name in DETECTION_SCHEMA.names if name != "schema_version"]
NAMES = ("camera_snapshots", "vehicle_detections", "camera_metadata")
PRIMARY_KEYS = {NAMES[0]: KEY, NAMES[1]: KEY + ["detection_index"],
                NAMES[2]: ["camera_id"]}


def _schema(fields: list[pa.Field], name: str) -> pa.Schema:
    return pa.schema(fields, metadata={
        b"contract_version": CONTRACT.encode(), b"table": name.encode(),
        b"capture_time_policy": b"naive filename clock; timezone and accuracy unverified",
    })


SCHEMAS = {
    NAMES[0]: _schema([IMAGE_SCHEMA.field(c).with_nullable(c == "timestamp")
                      for c in SNAPSHOT_FIELDS], NAMES[0]),
    NAMES[1]: _schema([DETECTION_SCHEMA.field(c).with_nullable(c == "timestamp")
                      for c in DETECTION_FIELDS], NAMES[1]),
    NAMES[2]: _schema([
        pa.field(c, pa.float64() if c in ("latitude", "longitude") else
                 pa.int64() if c == "image_count" else pa.string(),
                 nullable=c in {"location_detail", "location_en", "location_detail_en", "ip_address"})
        for c in CAMERA_FIELDS], NAMES[2]),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _project(frame: pd.DataFrame, name: str) -> pa.Table:
    schema = SCHEMAS[name]
    ordered = frame.sort_values(PRIMARY_KEYS[name], kind="stable").reset_index(drop=True)
    table = pa.Table.from_pandas(ordered[schema.names], schema=schema,
                                preserve_index=False, safe=True).replace_schema_metadata(schema.metadata)
    for field in schema:
        _require(field.nullable or table[field.name].null_count == 0,
                 f"{name}.{field.name} violates non-null contract")
    return table


def build_exports(manifest: pd.DataFrame, images: pd.DataFrame,
                  detections: pd.DataFrame, cameras: pd.DataFrame,
                  run_id: str) -> dict[str, pa.Table]:
    """Validate existing values, normalize metadata and project explicit schemas."""
    checks = validate_tables(manifest, images, detections, run_id)
    errors = [c["category"] for c in checks.checks
              if c["severity"] == "critical" and c["status"] == "fail"]
    _require(not errors, f"Source structural validation failed: {sorted(set(errors))}")
    _require(set(CAMERA_FIELDS).issubset(cameras), "Camera CSV missing contract fields")
    _require(not cameras.camera_id.isna().any() and not cameras.camera_id.duplicated().any(),
             "Camera IDs must be non-null and unique")
    ids = cameras.camera_id.astype("string")
    _require(ids.eq(ids.str.strip()).all() and ids.ne("").all(),
             "Camera IDs must be nonblank existing strings without normalization")
    _require(set(manifest.camera_id) == set(cameras.camera_id),
             "Camera CSV and manifest must represent exactly the same cameras")
    # The manifest is the metadata snapshot used by engineering; refuse conflicts.
    lookup = cameras.set_index("camera_id")
    for field in CAMERA_FIELDS[1:]:
        _require(field in manifest, f"Manifest lacks metadata field: {field}")
        expected = manifest.camera_id.map(lookup[field])
        equal = manifest[field].eq(expected).fillna(False)
        equal |= manifest[field].isna() & expected.isna()
        _require(equal.all(), f"Camera CSV differs from manifest metadata: {field}")
    _require(np.isfinite(cameras[["latitude", "longitude"]].to_numpy(dtype=float)).all(),
             "Camera coordinates must be finite")
    _require(cameras.latitude.between(-90, 90).all()
             and cameras.longitude.between(-180, 180).all(), "Invalid camera coordinates")
    _require(cameras.image_count.ge(0).all(), "Metadata image_count must be nonnegative")
    successful = images.loc[images.status.eq("success")]
    tables = {NAMES[0]: _project(successful, NAMES[0]),
              NAMES[1]: _project(detections, NAMES[1]),
              NAMES[2]: _project(cameras, NAMES[2])}
    validate_exports(tables, successful, detections)
    return tables


def validate_exports(tables: dict[str, pa.Table], source_images: pd.DataFrame,
                     source_detections: pd.DataFrame) -> None:
    """Prove value preservation, keys, FKs, count and zero-snapshot invariants."""
    frames = {}
    for name in NAMES:
        table = tables[name]
        _require(table.schema.equals(SCHEMAS[name], check_metadata=True),
                 f"Invalid export schema: {name}")
        _require(all(f.nullable or table[f.name].null_count == 0 for f in table.schema),
                 f"Invalid nullability: {name}")
        frame = table.to_pandas()
        _require(not frame.duplicated(PRIMARY_KEYS[name]).any(), f"Duplicate keys: {name}")
        frames[name] = frame
    for name, source in [(NAMES[0], source_images), (NAMES[1], source_detections)]:
        _require(tables[name].equals(_project(source, name)),
                 f"Values or row counts changed during export: {name}")
    snapshots, detections, cameras = [frames[name] for name in NAMES]
    parent_keys = pd.MultiIndex.from_frame(snapshots[KEY])
    _require(pd.MultiIndex.from_frame(detections[KEY]).isin(parent_keys).all(),
             "Export contains orphan detections")
    _require(snapshots.camera_id.isin(cameras.camera_id).all(), "Unknown snapshot camera")
    joined = detections.merge(snapshots[KEY + ["camera_id", "timestamp", "image_path"]],
                              on=KEY, suffixes=("", "_parent"), validate="many_to_one")
    for field in ("camera_id", "timestamp", "image_path"):
        equal = joined[field].eq(joined[f"{field}_parent"]).fillna(False)
        equal |= joined[field].isna() & joined[f"{field}_parent"].isna()
        _require(equal.all(), f"Detection parent metadata mismatch: {field}")
    for label in ("car", "motorcycle", "bus", "truck", "total_vehicle"):
        chosen = detections if label == "total_vehicle" else detections.loc[detections.vehicle_class.eq(label)]
        counts = chosen.groupby(KEY).size().reindex(parent_keys, fill_value=0).to_numpy()
        _require(np.array_equal(counts, snapshots[f"{label}_count"]),
                 f"Export count mismatch: {label}")
    _require(int(snapshots.total_vehicle_count.eq(0).sum())
             == int(source_images.total_vehicle_count.eq(0).sum()), "Zero snapshots lost")


def _read_sources(root: Path, run: Path, manifest_path: Path,
                  validation_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, str]:
    """Require a current passed Step 6 report and intact committed checkpoints."""
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    _require(report.get("schema_version") == "detection-validation-1.0.0"
             and report.get("status") in {"passed", "passed_with_warnings"}
             and report.get("issue_counts", {}).get("critical") == 0
             and report.get("ready_for_step7_planning") is True,
             "Step 6 has not passed; export blocked")
    _require(report.get("provenance") == metadata, "Step 6 belongs to a different run")
    _require(not any(c.get("severity") == "critical" and c.get("status") != "pass"
                     for c in report.get("checks", [])), "Step 6 contains failed critical checks")
    proven = report.get("validation_provenance", {})
    for key, path in [("run_metadata_sha256", run / "run.json"),
                      ("summary_sha256", run / "summary.json"),
                      ("manifest_sha256", manifest_path)]:
        _require(proven.get(key) == sha256(path), f"Stale Step 6 report: {key}")
    _require(summary.get("status") == "complete" and summary.get("remaining_images") == 0,
             "Source run is not finalized")
    images, detections, markers = [], [], []
    cursor = 0
    for batch in sorted((run / "batches").glob("batch-*")):
        marker = json.loads((batch / "commit.json").read_text(encoding="utf-8"))
        start, stop = marker["start"], marker["stop"]
        _require(start == cursor and start < stop and marker["run_id"] == metadata["run_id"]
                 and batch.name == f"batch-{start:08d}-{stop:08d}", "Invalid checkpoint sequence")
        for filename, expected_schema, parts, count_key, hash_key in [
            ("image_counts.parquet", IMAGE_SCHEMA, images, "image_count", "image_sha256"),
            ("detections.parquet", DETECTION_SCHEMA, detections, "detection_count", "detection_sha256"),
        ]:
            path = batch / filename
            _require(sha256(path) == marker[hash_key], f"Checkpoint checksum failed: {path}")
            table = pq.read_table(path)
            _require(table.schema.equals(expected_schema) and len(table) == marker[count_key],
                     f"Checkpoint schema/count failed: {path}")
            parts.append(table)
        _require(marker["image_count"] == stop - start, "Checkpoint range/count differs")
        markers.append(marker)
        cursor = stop
    manifest = pq.read_table(manifest_path).to_pandas()
    _require(images and cursor == len(manifest) == metadata["target_images"]
             and not list((run / "batches").glob(".pending-*")), "Incomplete full run")
    image_frame, detection_frame = [pa.concat_tables(parts).to_pandas()
                                    for parts in (images, detections)]
    stats = report["statistics"]
    _require(len(image_frame) == stats["image_rows"] == summary["processed_images"]
             and len(detection_frame) == stats["detection_rows"] == summary["detection_count"]
             and len(markers) == stats["committed_batches"] == summary["committed_batches"],
             "Source row/checkpoint counts differ from Step 6/summary")
    marker_hash = hashlib.sha256(json.dumps(markers, sort_keys=True).encode()).hexdigest()
    return manifest, image_frame, detection_frame, metadata, report, marker_hash


def _publish_file(source: Path, target: Path) -> None:
    """Atomic exclusive publication on the same filesystem; never replace a file."""
    os.link(source, target)


def export_datasets(project_root: Path, run_dir: Path, manifest_path: Path,
                    camera_csv: Path, validation_report: Path, output_dir: Path) -> dict:
    """Publish checked tables then a commit manifest; identical reruns are no-ops."""
    root = project_root.resolve()
    run, manifest_path, camera_csv, validation_report, output = [
        (root / p).resolve() for p in
        (run_dir, manifest_path, camera_csv, validation_report, output_dir)]
    _require(output.is_relative_to((root / "data/processed").resolve()),
             "Export directory must be under data/processed")
    manifest, images, detections, metadata, report, marker_hash = _read_sources(
        root, run, manifest_path, validation_report)
    cameras = pd.read_csv(camera_csv, dtype={"camera_id": "string"})
    tables = build_exports(manifest, images, detections, cameras, metadata["run_id"])
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output), TemporaryDirectory(prefix=".export-", dir=output) as temporary:
        staging = Path(temporary)
        files = {}
        for name, table in tables.items():
            path = staging / f"{name}.parquet"
            with path.open("xb") as stream:
                pq.write_table(table, stream, compression="snappy", version="2.6")
                stream.flush()
                os.fsync(stream.fileno())
            # Validate the serialized data, not only the in-memory projection.
            _require(pq.read_table(path).equals(table, check_metadata=True),
                     f"Parquet roundtrip changed data: {name}")
            files[path.name] = {"rows": len(table), "unique_keys": len(table),
                                "bytes": path.stat().st_size, "sha256": sha256(path),
                                "primary_key": PRIMARY_KEYS[name]}
        provenance = {
            "contract_version": CONTRACT, "status": "complete", "source_run": metadata,
            "step6": {"report_sha256": sha256(validation_report),
                      "generated_at_utc": report["generated_at_utc"],
                      "status": report["status"], "issue_counts": report["issue_counts"],
                      "warnings": report["issues_by_category"]["warning"]},
            "input_sha256": {"manifest": sha256(manifest_path), "camera_csv": sha256(camera_csv),
                             "checkpoints": marker_hash, "exporter": sha256(Path(__file__))},
            "software": {p: version(p) for p in ("pandas", "pyarrow", "numpy")},
            "class_mapping": {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"},
            "timestamp_policy": "Unchanged naive filename clock readings; timezone/accuracy unverified",
            "bbox_policy": "Float64 pixel xyxy in Step 5 EXIF-normalized decoded-image dimensions",
            "interpretation": "Model-estimated vehicles visible per snapshot; not ground truth, traffic flow, throughput, speed or validated congestion",
            "excluded_failed_images": int(images.status.ne("success").sum()),
            "successful_zero_detection_images": int(images.loc[images.status.eq("success"), "total_vehicle_count"].eq(0).sum()),
            "files": files,
        }
        marker_path = staging / "export_manifest.json"
        with marker_path.open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Preflight ALL existing targets before publishing anything. Partial identical
        # outputs after interruption are reused; incompatible files are never replaced.
        planned = [staging / name for name in files] + [marker_path]
        for path in planned:
            target = output / path.name
            _require(not target.exists() or sha256(target) == sha256(path),
                     f"Existing incompatible output: {target}; choose a new output directory")
        for path in planned:
            target = output / path.name
            if not target.exists():
                _publish_file(path, target)
        read_exports(output)
    LOG.info("Export complete: %d snapshots, %d predictions, %d cameras",
             len(tables[NAMES[0]]), len(tables[NAMES[1]]), len(tables[NAMES[2]]))
    return provenance


def read_exports(output_dir: Path) -> dict[str, pa.Table]:
    """Consumer gate: require the last-published manifest, schemas and checksums."""
    marker = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
    _require(marker.get("contract_version") == CONTRACT and marker.get("status") == "complete",
             "Unsupported or incomplete observation export")
    _require(set(marker["files"]) == {f"{n}.parquet" for n in NAMES}, "Incomplete export manifest")
    result = {}
    for name in NAMES:
        path = output_dir / f"{name}.parquet"
        expected = marker["files"][path.name]
        _require(sha256(path) == expected["sha256"], f"Export checksum failed: {path}")
        table = pq.read_table(path)
        _require(table.schema.equals(SCHEMAS[name], check_metadata=True)
                 and len(table) == expected["rows"], f"Export schema/count failed: {name}")
        result[name] = table
    return result
