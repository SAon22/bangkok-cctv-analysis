"""Read-only Step 6 structural validation of committed Step 5 predictions."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .detection_checkpoints import sha256
from .pipeline import BASELINE, COUNTS, DETECTION_SCHEMA, IMAGE_SCHEMA, VERSION
from .vehicle_detector import COCO_VEHICLES
from .yolo_feasibility import validate_manifest

LOG = logging.getLogger(__name__)
ISSUE_SCHEMA = pa.schema([
    ("severity", pa.string()), ("category", pa.string()),
    ("table", pa.string()), ("row_index", pa.int64()),
    ("image_id", pa.string()), ("image_path", pa.string()),
    ("detection_index", pa.int64()), ("message", pa.string()),
])
KEY = ["run_id", "image_id"]
STATUSES = {"success", "missing_file", "invalid_path", "unreadable", "inference_error"}


class Findings:
    """Collect vectorized row masks and explicit dataset-level findings."""

    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.parts: list[pd.DataFrame] = []

    def check(self, category: str, message: str, *, severity: str = "critical",
              frame: pd.DataFrame | None = None, bad: pd.Series | None = None,
              failed: bool = False, table: str = "dataset") -> None:
        selected = frame.loc[bad.fillna(True)] if frame is not None and bad is not None else None
        count = len(selected) if selected is not None else int(failed)
        self.checks.append({"category": category, "severity": severity,
                            "status": "fail" if count else "pass", "issue_count": count,
                            "description": message})
        if not count:
            return
        issues = pd.DataFrame(index=range(count))
        issues["severity"], issues["category"] = severity, category
        issues["table"], issues["message"] = table, message
        issues["row_index"] = selected.index.to_numpy() if selected is not None else None
        for field in ("image_id", "image_path", "detection_index"):
            issues[field] = (selected[field].to_numpy() if selected is not None
                             and field in selected else None)
        self.parts.append(issues)

    def table(self) -> pa.Table:
        if not self.parts:
            return pa.Table.from_pylist([], schema=ISSUE_SCHEMA)
        return pa.Table.from_pandas(pd.concat(self.parts, ignore_index=True),
                                   schema=ISSUE_SCHEMA, preserve_index=False)


def _different(left: pd.Series, right: pd.Series) -> pd.Series:
    return ~(left.eq(right).fillna(False) | (left.isna() & right.isna()))


def validate_tables(manifest: pd.DataFrame, images: pd.DataFrame,
                    detections: pd.DataFrame, run_id: str,
                    findings: Findings | None = None) -> Findings:
    """Validate schema-shaped tables using vectorized masks; never alter inputs.

    Arrow schema checks belong to the checkpoint reader. Missing columns in direct
    calls are reported and block dependent checks rather than causing a KeyError.
    """
    f = findings if findings is not None else Findings()
    required = {"manifest": {"image_path", "camera_id", "timestamp"},
                "images": set(IMAGE_SCHEMA.names),
                "detections": set(DETECTION_SCHEMA.names)}
    tables = {"manifest": manifest, "images": images, "detections": detections}
    missing = False
    for name, frame in tables.items():
        absent = required[name] - set(frame.columns)
        f.check(f"{name}_columns", f"Missing required columns: {sorted(absent)}",
                failed=bool(absent), table=name)
        missing |= bool(absent)
    if missing:
        return f
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        f.check("manifest_identity", str(exc), failed=True, table="manifest")
        return f
    f.check("manifest_identity", "Manifest keys and timestamp semantics are valid")
    for name, frame, key in [("images", images, KEY),
                             ("detections", detections, KEY + ["detection_index"])]:
        required_fields = list(frame.columns) if name == "detections" else [
            "schema_version", "run_id", "image_id", "image_path", "camera_id",
            "source_index", "status", "runtime_seconds", "processed_at"]
        required_fields = [field for field in required_fields if field != "timestamp"]
        f.check(f"{name}_required_values", "Required values must not be null",
                frame=frame, bad=frame[required_fields].isna().any(axis=1), table=name)
        strings = ["schema_version", "run_id", "image_id", "image_path", "camera_id"]
        blank = pd.Series(False, index=frame.index)
        for field in strings:
            blank |= frame[field].astype("string").str.strip().eq("").fillna(True)
        f.check(f"{name}_blank_identity", "Identity fields must be nonblank strings",
                frame=frame, bad=blank, table=name)
        f.check(f"{name}_keys", "Primary keys must be unique", frame=frame,
                bad=frame.duplicated(key, keep=False), table=name)
        f.check(f"{name}_run_version", "Schema version and run ID must match run.json",
                frame=frame, bad=frame.run_id.ne(run_id) | frame.schema_version.ne(VERSION),
                table=name)
    f.check("image_path_keys", "One record per manifest image_path", frame=images,
            bad=images.image_path.duplicated(keep=False), table="images")
    for category, frame, bad, name in [
        ("missing_manifest_images", manifest, ~manifest.image_path.isin(images.image_path), "manifest"),
        ("unexpected_images", images, ~images.image_path.isin(manifest.image_path), "images"),
    ]:
        f.check(category, "Image coverage must exactly match the manifest", frame=frame,
                bad=bad, table=name)
    # Hash only relative path strings, never source image bytes.
    expected_ids = images.image_path.fillna("").map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    f.check("image_id", "image_id must be SHA-256 of the UTF-8 relative image_path",
            frame=images, bad=images.image_id.ne(expected_ids), table="images")
    f.check("portable_image_paths", "Image paths must be portable project-relative POSIX paths",
            frame=images, bad=images.image_path.astype("string").str.contains(
                r"(?:^/|\\|^[A-Za-z]:|(?:^|/)\.\.(?:/|$))", regex=True).fillna(True), table="images")
    reference = manifest.drop_duplicates("image_path").set_index("image_path")
    for field in ("camera_id", "timestamp"):
        expected = images.image_path.map(reference[field])
        f.check(f"manifest_{field}", f"Image {field} must equal the manifest value",
                frame=images, bad=_different(images[field], expected), table="images")
    success = images.status.eq("success")
    f.check("processing_status", "Only finalized processing statuses are supported",
            frame=images, bad=~images.status.isin(STATUSES), table="images")
    f.check("success_counts", "Success requires non-null nonnegative counts",
            frame=images, bad=success & (images[COUNTS].isna().any(axis=1)
                                        | images[COUNTS].lt(0).any(axis=1)), table="images")
    f.check("failure_counts", "Failure counts must all be null", frame=images,
            bad=~success & images[COUNTS].notna().any(axis=1), table="images")
    f.check("error_messages", "Success has no error; failures need an explanation",
            frame=images, bad=(success & images.error_message.notna()) | (
                ~success & images.error_message.fillna("").str.strip().eq("")), table="images")
    dims = images[["width", "height"]]
    decoded = success | images.status.eq("inference_error")
    f.check("image_dimensions", "Success needs positive dimensions; optional failure dimensions must be positive",
            frame=images, bad=(decoded & dims.isna().any(axis=1)) | dims.le(0).any(axis=1), table="images")
    hashes = images.source_sha256.astype("string")
    f.check("source_hash_format", "Source hash must be 64 lowercase hex characters when present; required for success",
            frame=images, bad=(hashes.notna() & ~hashes.str.fullmatch(r"[0-9a-f]{64}").fillna(False))
            | (decoded & hashes.isna()), table="images")
    before_decode = images.status.isin(["missing_file", "invalid_path"])
    f.check("failure_stage_fields", "Pre-decode failures cannot have source/dimension/inference fields; unreadable images cannot have inference timing",
            frame=images, bad=(before_decode & images[["source_sha256", "width", "height", "inference_seconds"]].notna().any(axis=1))
            | (images.status.eq("unreadable") & images.inference_seconds.notna()), table="images")
    # The same immutable path must never carry conflicting hashes, even in a broken duplicate input.
    hash_counts = images.groupby("image_path", dropna=False).source_sha256.nunique()
    f.check("source_hash_consistency", "Repeated source paths must agree on source_sha256",
            frame=images, bad=images.image_path.map(hash_counts).gt(1), table="images")
    for field in ("runtime_seconds", "inference_seconds"):
        values = images[field].to_numpy(dtype=float, na_value=np.nan)
        bad = pd.Series((~np.isfinite(values) | (values < 0)), index=images.index)
        if field == "inference_seconds":
            bad &= images[field].notna() | success
        f.check(field, "Recorded durations must be finite and nonnegative (inference may be null if not attempted)",
                frame=images, bad=bad, table="images")
    f.check("supported_classes", "COCO class ID/name mapping must be 2=car, 3=motorcycle, 5=bus, 7=truck",
            frame=detections, bad=~detections.class_id.isin(COCO_VEHICLES)
            | _different(detections.vehicle_class, detections.class_id.map(COCO_VEHICLES)), table="detections")
    values = detections.confidence.to_numpy(dtype=float, na_value=np.nan)
    f.check("confidence", "Confidence must be finite and within [0,1]",
            frame=detections, bad=pd.Series(~np.isfinite(values) | (values < 0) | (values > 1),
                                            index=detections.index), table="detections")
    ordinals = detections.detection_index
    f.check("detection_ordinals", "Detection indices must be nonnegative and contiguous from zero per image",
            frame=detections, bad=ordinals.lt(0) | ordinals.ne(
                detections.groupby(KEY, dropna=False).detection_index.rank(method="first").sub(1)), table="detections")
    # Deduplicate only the lookup used for safe diagnostics; duplicates themselves remain critical issues.
    parents = images.drop_duplicates(KEY).set_index(KEY)
    joined = detections.merge(
        parents[["image_path", "camera_id", "timestamp", "status", "width", "height"]],
        left_on=KEY, right_index=True, how="left", suffixes=("", "_parent"), sort=False)
    f.check("detection_parent", "Every prediction must reference a successful image record",
            frame=detections, bad=joined.status.ne("success"), table="detections")
    for field in ("image_path", "camera_id", "timestamp"):
        f.check(f"detection_{field}", f"Detection {field} must equal its image record",
                frame=detections, bad=_different(joined[field], joined[f"{field}_parent"]), table="detections")
    boxes = detections[["x_min", "y_min", "x_max", "y_max"]]
    bad_box = ~np.isfinite(boxes.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    bad_box |= (boxes.x_min.lt(0) | boxes.y_min.lt(0) | boxes.x_min.ge(boxes.x_max)
                | boxes.y_min.ge(boxes.y_max) | boxes.x_max.gt(joined.width)
                | boxes.y_max.gt(joined.height) | joined.width.isna() | joined.height.isna()).to_numpy()
    f.check("bounding_boxes", "Boxes must be finite, positive-area and within recorded original dimensions",
            frame=detections, bad=pd.Series(bad_box, index=detections.index), table="detections")
    grouped = detections.groupby(KEY + ["vehicle_class"], dropna=False).size().unstack(fill_value=0)
    aligned = grouped.reindex(pd.MultiIndex.from_frame(images[KEY]), fill_value=0)
    mismatch = pd.Series(False, index=images.index)
    for name in COCO_VEHICLES.values():
        actual = aligned[name].to_numpy() if name in aligned else np.zeros(len(images), dtype=int)
        mismatch |= success & images[f"{name}_count"].ne(actual)
    actual_total = detections.groupby(KEY, dropna=False).size().reindex(
        pd.MultiIndex.from_frame(images[KEY]), fill_value=0).to_numpy()
    mismatch |= success & images.total_vehicle_count.ne(actual_total)
    mismatch |= success & images[COUNTS[:-1]].sum(axis=1).ne(images.total_vehicle_count)
    f.check("count_consistency", "Per-class and total image counts must equal detection records, including successful zeros",
            frame=images, bad=mismatch, table="images")
    f.check("processing_failures", "Failed images require investigation before downstream use",
            frame=images, bad=~success, table="images", severity="warning")
    f.check("extreme_counts", "More than 100 predictions per snapshot: review only, not automatically invalid",
            frame=images, bad=success & images.total_vehicle_count.gt(100), table="images", severity="warning")
    return f


def validate_run(project_root: Path, run_dir: Path, manifest_path: Path,
                 output_dir: Path) -> dict:
    """Read each checkpoint table once, validate, and write a fresh report directory."""
    tick = time.perf_counter()
    root = project_root.resolve()
    run_dir, manifest_path, output_dir = [(root / p).resolve()
                                        for p in (run_dir, manifest_path, output_dir)]
    interim = (root / "data/interim").resolve()
    if (not output_dir.is_relative_to(interim) or output_dir == interim
            or output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir)):
        raise ValueError("Choose a separate report subdirectory under data/interim")
    if output_dir.exists():
        raise FileExistsError(f"Reports already exist: {output_dir}; choose --output-dir with a new name")
    f = Findings()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_table = pq.read_table(manifest_path)
    manifest = manifest_table.to_pandas()
    signature = metadata.get("signature", {})
    f.check("manifest_fingerprint", "Manifest SHA-256 must match immutable run provenance",
            failed=sha256(manifest_path) != signature.get("manifest_sha256"))
    f.check("run_metadata", "Run version, baseline and full-dataset target must match the saved contract",
            failed=metadata.get("schema_version") != VERSION or signature.get("schema_version") != VERSION
            or signature.get("baseline") != BASELINE or signature.get("image_limit") is not None
            or metadata.get("target_images") != len(manifest) or not metadata.get("run_id"))
    valid_manifest_schema = all(field in manifest_table.column_names for field in ("camera_id", "image_path", "timestamp"))
    if valid_manifest_schema:
        valid_manifest_schema = (manifest_table.schema.field("timestamp").type == pa.timestamp("us")
            and all(pa.types.is_string(manifest_table.schema.field(c).type)
                    or pa.types.is_large_string(manifest_table.schema.field(c).type)
                    for c in ("camera_id", "image_path")))
    for field in ("manifest_sha256", "weights_sha256"):
        value = signature.get(field)
        f.check("provenance_hashes", f"Run signature {field} must be a SHA-256 hex digest",
                failed=not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value))
    f.check("manifest_schema", "Manifest needs string identity and naive timestamp[us]", failed=not valid_manifest_schema)
    image_parts, detection_parts = [], []
    cursor, readable = 0, True
    batches = sorted((run_dir / "batches").glob("batch-*"))
    for batch in batches:
        try:
            marker = json.loads((batch / "commit.json").read_text(encoding="utf-8"))
            start, stop = marker["start"], marker["stop"]
            f.check("checkpoint_sequence", f"Contiguous committed range: {batch.name}",
                    failed=marker["run_id"] != metadata.get("run_id") or start != cursor
                    or not start < stop <= len(manifest)
                    or batch.name != f"batch-{start:08d}-{stop:08d}")
            batch_size = signature.get("batch_size")
            f.check("checkpoint_configuration", f"Batch size and duration agree with run provenance: {batch.name}",
                    failed=not isinstance(batch_size, int) or batch_size < 1
                    or stop - start != min(batch_size, len(manifest) - start)
                    or not isinstance(marker.get("processing_seconds"), (int, float))
                    or not np.isfinite(marker.get("processing_seconds", float("nan")))
                    or marker.get("processing_seconds", -1) < 0)
            cursor = stop
            paths = [batch / "image_counts.parquet", batch / "detections.parquet"]
            tables = [pq.read_table(path) for path in paths]
            good_schema = tables[0].schema.equals(IMAGE_SCHEMA) and tables[1].schema.equals(DETECTION_SCHEMA)
            f.check("checkpoint_schema", f"Explicit Arrow schemas: {batch.name}", failed=not good_schema)
            f.check("checkpoint_checksum", f"Both committed file hashes: {batch.name}",
                    failed=sha256(paths[0]) != marker["image_sha256"] or sha256(paths[1]) != marker["detection_sha256"])
            f.check("checkpoint_counts", f"Commit row counts: {batch.name}",
                    failed=len(tables[0]) != stop - start or len(tables[0]) != marker["image_count"]
                    or len(tables[1]) != marker["detection_count"])
            if good_schema:
                image_parts.append(tables[0])
                detection_parts.append(tables[1])
            else:
                readable = False
        except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as exc:
            f.check("checkpoint_read", f"Cannot validate {batch.name}: {exc}", failed=True)
            readable = False
    f.check("finalized_checkpoints", "All manifest rows must be committed with no pending directories",
            failed=cursor != len(manifest) or not batches
            or bool(list((run_dir / "batches").glob(".pending-*"))))
    stats: dict = {"manifest_images": len(manifest), "committed_batches": len(batches)}
    if readable and image_parts and valid_manifest_schema:
        images = pa.concat_tables(image_parts).to_pandas()
        detections = pa.concat_tables(detection_parts).to_pandas()
        validate_tables(manifest, images, detections, metadata.get("run_id"), f)
        plan = manifest.sort_values(["camera_id", "timestamp", "image_path"], kind="stable", na_position="last")
        f.check("source_order", "Source indices and image paths match the deterministic full manifest plan",
                failed=len(images) != len(plan) or not np.array_equal(images.source_index, np.arange(len(images)))
                or images.image_path.tolist() != plan.image_path.tolist())
        threshold = signature.get("baseline", {}).get("confidence", .25)
        f.check("configured_threshold", "Predictions must meet saved confidence threshold",
                frame=detections, bad=detections.confidence.lt(threshold), table="detections")
        stats.update({"image_rows": len(images), "detection_rows": len(detections),
                      "unique_image_keys": len(images.drop_duplicates(KEY)),
                      "unique_detection_keys": len(detections.drop_duplicates(KEY + ["detection_index"])),
                      "duplicate_image_key_rows": int(images.duplicated(KEY, keep=False).sum()),
                      "duplicate_detection_key_rows": int(detections.duplicated(KEY + ["detection_index"], keep=False).sum()),
                      "missing_manifest_images": int((~manifest.image_path.isin(images.image_path)).sum()),
                      "unexpected_images": int((~images.image_path.isin(manifest.image_path)).sum()),
                      "max_snapshot_vehicle_count": int(images.total_vehicle_count.max()) if images.total_vehicle_count.notna().any() else None,
                      "cameras": int(images.camera_id.nunique()),
                      "successful_images": int(images.status.eq("success").sum()),
                      "failed_images": int(images.status.ne("success").sum()),
                      "zero_detection_images": int((images.status.eq("success") & images.total_vehicle_count.eq(0)).sum()),
                      "class_distribution": {name: int(detections.vehicle_class.eq(name).sum()) for name in COCO_VEHICLES.values()},
                      "missing_values": {name: {c: int(n) for c, n in frame.isna().sum().items()}
                                         for name, frame in [("manifest", manifest), ("images", images), ("detections", detections)]}})
        expected_summary = {"schema_version": VERSION, "run_id": metadata.get("run_id"), "status": "complete",
                            "target_images": len(manifest), "processed_images": len(images), "remaining_images": 0,
                            "committed_batches": len(batches), "detection_count": len(detections),
                            "successful_zero_detection_images": stats["zero_detection_images"],
                            "status_counts": {str(k): int(v) for k, v in images.status.value_counts().items()},
                            "per_class_detection_counts": stats["class_distribution"]}
        f.check("final_summary", "Final summary fields must agree with committed data",
                failed=any(summary.get(k) != v for k, v in expected_summary.items()))
        optional = [c for c in ("location_en", "location_detail_en") if c in manifest]
        f.check("optional_metadata", "Optional English metadata contains nulls; see missing_values",
                failed=bool(optional) and manifest[optional].isna().any().any(), severity="warning")
        core = [c for c in ("location", "latitude", "longitude", "metadata_matched") if c in manifest]
        bad_metadata = manifest[core].isna().any(axis=1)
        if "metadata_matched" in manifest:
            bad_metadata |= manifest.metadata_matched.ne(True)
        f.check("camera_metadata", "Missing core camera metadata or unmatched camera join requires review",
                frame=manifest, bad=bad_metadata, table="manifest", severity="warning")
        class_counts = stats["class_distribution"]
        f.check("class_distribution_review", "A class exceeds 90% of predictions; review without removing predictions",
                failed=len(detections) > 0 and max(class_counts.values()) / len(detections) > .9, severity="warning")
    f.check("model_accuracy_limitations", "Pretrained predictions have known missed/incorrect/overlapping detections; structural validity does not establish accuracy or traffic flow",
            failed=True, severity="warning")
    f.check("timestamp_provenance", "Capture timestamps are unchanged naive filename readings; timezone/clock accuracy remains unverified",
            failed=True, severity="warning")
    issues = f.table()
    issue_frame = issues.to_pandas()
    counts = {severity: int(issue_frame.severity.eq(severity).sum()) for severity in ("critical", "warning")}
    by_category = {severity: {str(k): int(v) for k, v in issue_frame.loc[
        issue_frame.severity.eq(severity), "category"].value_counts().items()} for severity in counts}
    ready = counts["critical"] == 0
    report = {"schema_version": "detection-validation-1.0.0",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed_with_warnings" if ready and counts["warning"] else ("passed" if ready else "failed"),
              "input_run": run_dir.relative_to(root).as_posix() if run_dir.is_relative_to(root) else str(run_dir),
              "input_manifest": str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
              "provenance": metadata,
              "validation_provenance": {"validator_sha256": sha256(Path(__file__)),
                                        "run_metadata_sha256": sha256(run_dir / "run.json"),
                                        "summary_sha256": sha256(run_dir / "summary.json"),
                                        "manifest_sha256": sha256(manifest_path)},
              "statistics": stats,
              "issue_counts": counts, "issues_by_category": by_category,
              "checks": f.checks, "validation_seconds": time.perf_counter() - tick,
              "ready_for_step7_planning": ready,
              "readiness_recommendation": ("Structurally ready for Step 7 contract discussion; approve timezone, schema and consumer semantics separately. No integration implemented."
                                            if ready else "Blocked: resolve critical findings before Step 7."),
              "correction_policy": "No predictions removed or modified. Critical issue rows are proposed investigation targets, not automatic corrections.",
              "source_hash_policy": "Validate recorded formats/consistency; no raw image bytes read or rehashed.",
              "limits": ["No image pixels read; dimensions rely on recorded Step 5 decode metadata.",
                         "Structural checks do not estimate precision, recall or label correctness."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "data_quality_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    pq.write_table(issues, output_dir / "validation_issues.parquet")
    lines = ["# Step 6 detection data quality report", "", f"Status: **{report['status']}**", "",
             report["readiness_recommendation"], "", "## Dataset", "",
             "```json", json.dumps({k: v for k, v in stats.items() if k != "missing_values"}, indent=2), "```", "",
             "## Issues by category", "", "```json", json.dumps(by_category, indent=2), "```", "",
             "## Validation checks", "", "| Check | Severity | Status | Issues |", "| --- | --- | --- | ---: |"]
    # Collapse repeated per-checkpoint pass records for a concise human report.
    checks = pd.DataFrame(f.checks)
    for (category, severity), group in checks.groupby(["category", "severity"], sort=True):
        count = int(group.issue_count.sum())
        lines.append(f"| {category} ({len(group)} checks) | {severity} | {'fail' if count else 'pass'} | {count} |")
    lines += ["", "## Provenance and limitations", "", f"Run ID: `{metadata.get('run_id')}`.",
              f"Baseline: `{json.dumps(signature.get('baseline'))}`.",
              "Full input provenance, missing-value counts and individual checks are in the JSON report.",
              "Known warnings and individual affected rows are in validation_issues.parquet.",
              "No source data or predictions were modified. No raw images were read or rehashed.",
              "Recorded dimensions were checked, not independently decoded again.",
              "Model predictions are not ground truth, unique traffic volume or congestion.",
              "Timezone/clock provenance remains unverified; timestamps were not converted.",
              "Step 7 has not been implemented."]
    (output_dir / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Validation %s: %d critical, %d warning issue records", report["status"], counts["critical"], counts["warning"])
    return report
