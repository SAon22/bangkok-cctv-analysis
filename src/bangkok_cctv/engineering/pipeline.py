"""Step 5 raw detection pipeline; cleaning and integration remain separate."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import platform
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from .detection_checkpoints import atomic_json, commit_batch, recover_batches, sha256, writer_lock
from .vehicle_detector import COCO_VEHICLES, VehiclePredictor, YOLOVehicleDetector
from .yolo_feasibility import resolve_image_path, validate_manifest

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
VERSION = "detection-raw-1.0.0"
BASELINE = {"model": "yolov8n", "confidence": .25, "image_size": 640,
            "iou": .7, "device": "cpu", "max_detections": 300,
            "classes": [2, 3, 5, 7], "augment": False}
IDENTITY = [
    ("schema_version", pa.string()), ("run_id", pa.string()),
    ("image_id", pa.string()), ("image_path", pa.string()),
    ("camera_id", pa.string()), ("timestamp", pa.timestamp("us")),
]
COUNTS = [f"{name}_count" for name in COCO_VEHICLES.values()] + ["total_vehicle_count"]
IMAGE_SCHEMA = pa.schema(IDENTITY + [
    ("source_index", pa.int64()), ("status", pa.string()), ("error_message", pa.string()),
    ("source_sha256", pa.string()), ("width", pa.int64()), ("height", pa.int64()),
] + [(field, pa.int64()) for field in COUNTS] + [
    ("runtime_seconds", pa.float64()), ("inference_seconds", pa.float64()),
    ("processed_at", pa.timestamp("us", tz="UTC")),
])
DETECTION_SCHEMA = pa.schema(IDENTITY + [
    ("detection_index", pa.int64()), ("class_id", pa.int64()),
    ("vehicle_class", pa.string()), ("confidence", pa.float64()),
    ("x_min", pa.float64()), ("y_min", pa.float64()),
    ("x_max", pa.float64()), ("y_max", pa.float64()),
])


def process_image(
    root: Path, row: dict, source_index: int, run_id: str, predictor: VehiclePredictor,
) -> tuple[dict, list[dict]]:
    """One isolated attempt. Failed counts are null, never zero."""
    tick = time.perf_counter()
    identity = {
        "schema_version": VERSION, "run_id": run_id,
        "image_id": hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest(),
        **row,
    }
    record = {
        **identity, "source_index": source_index, "status": "pending",
        "error_message": None, "source_sha256": None, "width": None, "height": None,
        **{field: None for field in COUNTS}, "inference_seconds": None,
    }
    detections = []
    stage = "invalid_path"
    image = None
    try:
        path = resolve_image_path(root, row["image_path"])
        stage = "unreadable"
        content = path.read_bytes()
        record["source_sha256"] = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        record["width"], record["height"] = image.size
        stage = "inference_error"
        inference_start = time.perf_counter()
        try:
            predictions = predictor.predict(image)
        finally:
            record["inference_seconds"] = time.perf_counter() - inference_start
        for index, prediction in enumerate(predictions):
            prediction.validate(*image.size)
            if prediction.confidence < BASELINE["confidence"]:
                raise ValueError("Detector returned a below-threshold prediction")
            detections.append({
                **identity, "detection_index": index, "class_id": prediction.class_id,
                "vehicle_class": prediction.vehicle_class, "confidence": prediction.confidence,
                "x_min": prediction.x_min, "y_min": prediction.y_min,
                "x_max": prediction.x_max, "y_max": prediction.y_max,
            })
        for class_id, name in COCO_VEHICLES.items():
            record[f"{name}_count"] = sum(item.class_id == class_id for item in predictions)
        record["total_vehicle_count"] = len(detections)
        record["status"] = "success"
    except FileNotFoundError as exc:
        record["status"] = "missing_file" if stage == "invalid_path" else stage
        record["error_message"] = f"{type(exc).__name__}: {exc}"
        detections = []
    except Exception as exc:
        record["status"], record["error_message"] = stage, f"{type(exc).__name__}: {exc}"
        detections = []
        LOG.warning("%s: %s", row["image_path"], record["error_message"])
    finally:
        if record["status"] != "success":
            record.update({field: None for field in COUNTS})
        if image is not None:
            image.close()
        record["runtime_seconds"] = time.perf_counter() - tick
        record["processed_at"] = datetime.now(timezone.utc)
    return record, detections


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ["ultralytics", "torch", "torchvision", "Pillow", "numpy", "pandas", "pyarrow"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def summarize(output: Path, metadata: dict, commits: list[dict],
              invocation_start: float, model_load_seconds: float) -> dict:
    """Summary is a replaceable cache; committed Parquet directories are authoritative."""
    statuses: dict[str, int] = {}
    class_counts = {name: 0 for name in COCO_VEHICLES.values()}
    zeros, processed, inference_seconds, image_seconds = 0, 0, 0., 0.
    for marker in commits:
        batch = output / "batches" / f"batch-{marker['start']:08d}-{marker['stop']:08d}"
        rows = pq.read_table(batch / "image_counts.parquet").to_pylist()
        for row in rows:
            processed += 1
            inference_seconds += row["inference_seconds"] or 0.
            image_seconds += row["runtime_seconds"]
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
            if row["status"] == "success":
                zeros += row["total_vehicle_count"] == 0
                for name in class_counts:
                    class_counts[name] += row[f"{name}_count"]
    result = {
        "schema_version": VERSION, "run_id": metadata["run_id"],
        "status": "complete" if processed == metadata["target_images"] else "in_progress",
        "target_images": metadata["target_images"], "processed_images": processed,
        "remaining_images": metadata["target_images"] - processed,
        "committed_batches": len(commits), "status_counts": statuses,
        "successful_zero_detection_images": zeros, "per_class_detection_counts": class_counts,
        "detection_count": sum(item["detection_count"] for item in commits),
        "checkpoint_processing_seconds": sum(item["processing_seconds"] for item in commits),
        "total_inference_seconds": inference_seconds,
        "total_image_runtime_seconds": image_seconds,
        "last_invocation_model_load_seconds": model_load_seconds,
        "last_invocation_seconds_before_summary_write": time.perf_counter() - invocation_start,
        "note": "Raw predictions, not ground truth, traffic flow or congestion.",
    }
    atomic_json(output / "summary.json", result)
    return result


def run_detection_pipeline(
    project_root: Path, manifest_path: Path, output_dir: Path, weights_path: Path,
    *, batch_size: int = 100, image_limit: int | None = None,
    resume: bool = False, predictor: VehiclePredictor | None = None,
    max_batches: int | None = None,
) -> dict:
    """Commit serial image inference in bounded batches; resume identical plans only."""
    if batch_size < 1 or (image_limit is not None and image_limit < 1):
        raise ValueError("batch_size and image_limit must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    invocation_start = time.perf_counter()
    model_load_seconds = 0.
    root = project_root.resolve()
    manifest_path, weights_path = (root / manifest_path).resolve(), (root / weights_path).resolve()
    output = (root / output_dir).resolve()
    if output == root / "data/interim" or not output.is_relative_to((root / "data/interim").resolve()):
        raise ValueError("Output must be a run subdirectory under data/interim")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Local YOLOv8n weights required: {weights_path}; no downloads")
    reference = root / "data/interim/yolo_feasibility/summary.json"
    if reference.is_file():
        prior = json.loads(reference.read_text(encoding="utf-8"))
        if (prior.get("model_identifier") != "yolov8n.pt"
                or sha256(weights_path) != prior.get("weights_sha256")):
            raise ValueError("Weights do not match the trusted YOLOv8n feasibility baseline")
    elif weights_path.name != "yolov8n.pt":
        raise ValueError("Expected local yolov8n.pt baseline weights")
    frame = pd.read_parquet(manifest_path)
    validate_manifest(frame)
    frame = frame[["image_path", "camera_id", "timestamp"]].sort_values(
        ["camera_id", "timestamp", "image_path"], kind="stable", na_position="last"
    )
    if image_limit is not None:
        frame = frame.head(image_limit)
    if pq.read_schema(manifest_path).field("timestamp").type != pa.timestamp("us"):
        raise ValueError("Expected manifest timestamp[us] without timezone; no implicit conversion")
    plan = pa.Table.from_pandas(frame, preserve_index=False).to_pylist()
    signature = {
        "schema_version": VERSION, "manifest_sha256": sha256(manifest_path),
        "weights_sha256": sha256(weights_path), "baseline": BASELINE,
        "image_limit": image_limit, "batch_size": batch_size,
        "versions": package_versions(),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["pipeline.py", "detection_checkpoints.py", "vehicle_detector.py", "yolo_feasibility.py"]
        },
        "predictor": "ultralytics" if predictor is None else "injected-test-predictor",
    }
    if resume:
        if not (output / "run.json").is_file():
            raise ValueError("Cannot resume: run.json missing; choose a fresh run directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with writer_lock(output):
        if resume:
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            if metadata["signature"] != signature or metadata["target_images"] != len(plan):
                raise ValueError("Incompatible resume: manifest, model, parameters, versions or code differ")
        else:
            metadata = {
                "schema_version": VERSION, "run_id": uuid.uuid4().hex,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_path": (manifest_path.relative_to(root).as_posix()
                                  if manifest_path.is_relative_to(root) else str(manifest_path)),
                "weights_path": (weights_path.relative_to(root).as_posix()
                                 if weights_path.is_relative_to(root) else str(weights_path)),
                "platform": platform.platform(),
                "signature": signature, "target_images": len(plan),
                "order": ["camera_id (string)", "timestamp (null last)", "image_path"],
                "batch_size_semantics": "Checkpoint chunk size; inference is serial per image",
                "timestamp_policy": "Unchanged naive manifest clock values; no UTC conversion",
            }
            atomic_json(output / "run.json", metadata)
        (output / "batches").mkdir(exist_ok=True)
        cursor, commits = recover_batches(
            output / "batches", plan, metadata["run_id"], IMAGE_SCHEMA, DETECTION_SCHEMA
        )
        if cursor == len(plan):
            return summarize(output, metadata, commits, invocation_start, model_load_seconds)
        if predictor is None:
            load_start = time.perf_counter()
            predictor = YOLOVehicleDetector(
                weights_path, confidence=.25, image_size=640, device="cpu",
                iou=.7, max_detections=300,
            )
            model_load_seconds = time.perf_counter() - load_start
        for batch_number, start in enumerate(range(cursor, len(plan), batch_size), 1):
            images, detections = [], []
            tick = time.perf_counter()
            for index in range(start, min(start + batch_size, len(plan))):
                image, found = process_image(root, plan[index], index, metadata["run_id"], predictor)
                images.append(image)
                detections.extend(found)
            marker = commit_batch(
                output / "batches", start,
                pa.Table.from_pylist(images, schema=IMAGE_SCHEMA),
                pa.Table.from_pylist(detections, schema=DETECTION_SCHEMA),
                metadata["run_id"], time.perf_counter() - tick,
            )
            commits.append(marker)
            LOG.info("Committed %d/%d images; %d detections in this batch",
                     marker["stop"], len(plan), marker["detection_count"])
            if max_batches is not None and batch_number >= max_batches:
                break
        return summarize(output, metadata, commits, invocation_start, model_load_seconds)


def run_pipeline(config_path: Path, **overrides) -> dict:
    """Preserve the original config-path API, now implementing Step 5 only."""
    import yaml
    root = Path(overrides.pop("project_root", ROOT)).resolve()
    settings = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    config = settings.get("detection", {})
    options = {
        "manifest_path": Path(config.get("manifest", "data/interim/image_manifest.parquet")),
        "output_dir": Path(config.get("output_dir", "data/interim/detection_pipeline")),
        "weights_path": Path(settings.get("paths", {}).get("model_weights", "models/yolov8n.pt")),
        "batch_size": config.get("batch_size", 100),
        "image_limit": config.get("image_limit"), "resume": False,
    }
    options.update({key: value for key, value in overrides.items() if value is not None})
    return run_detection_pipeline(root, **options)
