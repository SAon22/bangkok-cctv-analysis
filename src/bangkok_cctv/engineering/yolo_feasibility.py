"""Deterministic, bounded CCTV sampling for experimental YOLO feasibility runs."""

from __future__ import annotations

import hashlib
import logging
import random
from collections import deque
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vehicle_detector import VehiclePredictor

import pandas as pd

LOG = logging.getLogger(__name__)
SCHEMA_VERSION = "feasibility-1.0.0"
TIME_BANDS = ("00-10", "11-15", "16-23")
REQUIRED_COLUMNS = {"image_path", "camera_id", "timestamp"}


def resolve_image_path(project_root: Path, image_path: str) -> Path:
    """Resolve a portable manifest path, rejecting absolute/escaping/missing files."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty relative string")
    relative = Path(image_path)
    if relative.is_absolute() or PureWindowsPath(image_path).drive:
        raise ValueError(f"Expected a project-relative image path: {image_path}")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image path escapes project root: {image_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    return resolved


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Check identity and time semantics without changing the source manifest."""
    if missing := REQUIRED_COLUMNS - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest has no images")
    for field in ("image_path", "camera_id"):
        if manifest[field].isna().any():
            raise ValueError(f"Manifest contains null {field}")
        if not manifest[field].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Manifest {field} must contain strings")
        if manifest[field].str.strip().eq("").any():
            raise ValueError(f"Manifest contains blank {field}")
    if manifest["image_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image paths")
    if not pd.api.types.is_datetime64_any_dtype(manifest["timestamp"]):
        raise ValueError("Manifest timestamp must be a pandas datetime column")
    if manifest["timestamp"].dt.tz is not None:
        raise ValueError("Expected timezone-naive manifest timestamps")


def select_representative_sample(
    manifest: pd.DataFrame, sample_size: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Choose one candidate per stratum, then round-robin across shuffled cameras.

    Input is sorted before seeded selection, so row order does not affect results.
    A small sample cannot cover every camera/date/band intersection.
    Invalid timestamps remain eligible in an explicit unknown stratum.
    """
    validate_manifest(manifest)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    reserved = {"image_id", "sample_date", "time_band", "schema_version"}
    if reserved & set(manifest.columns):
        raise ValueError(f"Manifest uses reserved fields: {sorted(reserved & set(manifest))}")
    frame = manifest.sort_values("image_path").reset_index(drop=True).copy()
    frame["sample_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d").fillna("unknown")
    frame["time_band"] = pd.cut(
        frame["timestamp"].dt.hour, [-1, 10, 15, 23], labels=TIME_BANDS
    ).astype("string").fillna("unknown")
    rng = random.Random(seed)
    candidates: dict[str, list[int]] = {}
    for (camera_id, _, _), group in frame.groupby(
        ["camera_id", "sample_date", "time_band"], observed=True, sort=True
    ):
        candidates.setdefault(str(camera_id), []).append(rng.choice(group.index.tolist()))
    queues: dict[str, deque[int]] = {}
    for camera_id, indices in candidates.items():
        rng.shuffle(indices)
        queues[camera_id] = deque(indices)
    camera_order = sorted(queues)
    rng.shuffle(camera_order)
    chosen = []
    while len(chosen) < sample_size and any(queues.values()):
        for camera_id in camera_order:
            if queues[camera_id] and len(chosen) < sample_size:
                chosen.append(queues[camera_id].popleft())
    if len(chosen) < sample_size:
        LOG.warning("Requested %d images; only %d nonempty strata exist",
                    sample_size, len(chosen))
    sample = frame.loc[chosen].sort_values(
        ["camera_id", "timestamp", "image_path"], na_position="last"
    ).reset_index(drop=True)
    sample["image_id"] = sample["image_path"].map(
        lambda path: hashlib.sha256(path.encode("utf-8")).hexdigest()
    )
    sample["schema_version"] = SCHEMA_VERSION
    return sample


# Feasibility-only artifact schema; intentionally separate from proposed production.
DETECTION_DTYPES = {
    "schema_version": "string", "model_run_id": "string", "image_id": "string",
    "image_path": "string", "camera_id": "string", "timestamp": "datetime64[us]",
    "detection_index": "int64", "class_id": "int64", "vehicle_class": "string",
    "confidence": "float64", "x_min": "float64", "y_min": "float64",
    "x_max": "float64", "y_max": "float64",
}
COUNT_FIELDS = (
    "car_count", "motorcycle_count", "bus_count", "truck_count",
    "total_vehicle_count",
)
LIMITATIONS = [
    "Pretrained COCO predictions are not ground truth; accuracy is not evaluated.",
    "Snapshot counts are not traffic flow, unique vehicles, or congestion.",
    "One candidate per camera/date/time-band stratum, then camera-balanced selection.",
    "Small samples cannot cover all intersections; rare/occluded vehicles may be missed.",
    "Sampling is reproducible for identical paths/times/seed, not an unbiased image census.",
    "Image IDs hash relative paths, not image contents; renaming changes identity.",
    "Timestamps remain timezone-naive; no UTC conversion or clock verification.",
    "Inference can vary with model/library/device versions; first-call warmup is included.",
]


def detection_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Construct a stable schema even when there are no detections."""
    return pd.DataFrame(records, columns=list(DETECTION_DTYPES)).astype(DETECTION_DTYPES)


def validate_outputs(sample: pd.DataFrame, detections: pd.DataFrame) -> None:
    """Enforce feasibility identity, foreign keys and count/failure semantics."""
    from .vehicle_detector import COCO_VEHICLES, VehicleDetection

    required = {
        "image_id", "image_path", "camera_id", "timestamp", "status",
        "schema_version", "model_run_id", "error_message", "width", "height",
        *COUNT_FIELDS,
    }
    if missing := required - set(sample.columns):
        raise ValueError(f"Sample output missing columns: {sorted(missing)}")
    if sample["image_id"].duplicated().any() or sample["image_path"].duplicated().any():
        raise ValueError("Sample identity must be unique")
    if set(detections.columns) != set(DETECTION_DTYPES):
        raise ValueError("Detection output has an unexpected schema")
    if detections.duplicated(["model_run_id", "image_id", "detection_index"]).any():
        raise ValueError("Detection primary keys must be unique")
    allowed_statuses = {
        "success", "skipped", "missing_file", "invalid_path",
        "unreadable", "inference_error",
    }
    if not sample["status"].isin(allowed_statuses).all():
        raise ValueError("Sample contains an invalid inference status")
    if not sample["schema_version"].eq(SCHEMA_VERSION).all():
        raise ValueError("Sample schema_version is unsupported")
    if sample[["image_id", "image_path", "camera_id", "model_run_id"]].isna().any().any():
        raise ValueError("Sample keys must be non-null")
    if sample["model_run_id"].nunique() != 1:
        raise ValueError("A feasibility artifact must contain exactly one model run")
    if sample["timestamp"].dt.tz is not None:
        raise ValueError("Sample timestamps must remain timezone-naive")
    success = sample["status"].eq("success")
    if sample.loc[success, "error_message"].notna().any():
        raise ValueError("Successful images cannot carry inference errors")
    if sample.loc[~success, "error_message"].isna().any():
        raise ValueError("Unsuccessful/skipped images must have an explanation")
    lookup = sample.set_index("image_id")
    for row in detections.itertuples(index=False):
        if row.image_id not in lookup.index:
            raise ValueError("Detection references an image outside the sample")
        parent = lookup.loc[row.image_id]
        if parent["status"] != "success":
            raise ValueError("Detection references an unsuccessful image")
        for field in ("image_path", "camera_id", "model_run_id", "schema_version"):
            if getattr(row, field) != parent[field]:
                raise ValueError(f"Detection {field} differs from its image")
        same_timestamp = (
            pd.isna(row.timestamp) and pd.isna(parent["timestamp"])
        ) or row.timestamp == parent["timestamp"]
        if not same_timestamp:
            raise ValueError("Detection timestamp differs from its image")
        if row.vehicle_class != COCO_VEHICLES.get(row.class_id):
            raise ValueError("Invalid detection class mapping")
        VehicleDetection(
            row.class_id, row.confidence, row.x_min, row.y_min, row.x_max, row.y_max
        ).validate(int(parent["width"]), int(parent["height"]))
    for row in sample.itertuples(index=False):
        subset = detections.loc[detections["image_id"].eq(row.image_id)]
        if row.status != "success":
            if not subset.empty or any(pd.notna(getattr(row, f)) for f in COUNT_FIELDS):
                raise ValueError("Unsuccessful images must have null counts and no detections")
            continue
        expected = {
            f"{name}_count": int(subset["class_id"].eq(class_id).sum())
            for class_id, name in COCO_VEHICLES.items()
        }
        expected["total_vehicle_count"] = sum(expected.values())
        if any(pd.isna(getattr(row, f)) or getattr(row, f) != n
               for f, n in expected.items()):
            raise ValueError("Image counts disagree with detections")


def _coverage(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "images": len(frame),
        "cameras": int(frame["camera_id"].nunique()),
        "dates": int(frame["timestamp"].dt.normalize().nunique()),
        "images_per_camera": {
            str(key): int(value) for key, value in frame["camera_id"].value_counts().items()
        },
        "images_per_date": {
            str(key): int(value) for key, value in
            frame["timestamp"].dt.strftime("%Y-%m-%d").fillna("unknown").value_counts().items()
        },
        "images_per_time_band": {
            str(key): int(value) for key, value in frame["time_band"].value_counts().items()
        },
    }


def run_feasibility(
    project_root: Path, output_dir: Path, *, sample_size: int = 100,
    seed: int = 42, sampling_only: bool = False,
    weights_path: Path | None = None, confidence: float = 0.25,
    image_size: int = 640, device: str = "cpu", iou: float = 0.7,
    max_detections: int = 300, download_weights: bool = False,
    predictor: VehiclePredictor | None = None,
) -> dict[str, object]:
    """Run a bounded experiment, persisting successes, zero counts and failures.

    predictor is an injected test adapter. CLI runs always use the real detector.
    Existing nonempty output directories are refused to preserve earlier evidence.
    """
    import hashlib
    import json
    import math
    import time
    import uuid
    from datetime import datetime, timezone
    from importlib.metadata import PackageNotFoundError, version

    from PIL import Image, ImageDraw

    from .vehicle_detector import (
        COCO_VEHICLES,
        YOLOVehicleDetector,
        download_default_weights,
    )

    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not math.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("iou must be between 0 and 1")
    if image_size < 32 or image_size % 32 or max_detections < 1:
        raise ValueError("image_size must be a multiple of 32; max_detections must be positive")
    if sampling_only and predictor is not None:
        raise ValueError("sampling_only cannot use a predictor")
    started = time.perf_counter()
    root = project_root.resolve()
    output_dir = (root / output_dir).resolve()
    interim = (root / "data/interim").resolve()
    if output_dir == interim or not output_dir.is_relative_to(interim):
        raise ValueError("Output must be a subdirectory of data/interim")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Choose a new --output-dir."
        )
    manifest_path = root / "data/interim/image_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    sample = select_representative_sample(manifest, sample_size, seed)
    sample["model_run_id"] = uuid.uuid4().hex
    sample["status"] = "pending"
    sample["error_message"] = pd.Series(pd.NA, index=sample.index, dtype="string")
    sample["annotation_error"] = pd.Series(pd.NA, index=sample.index, dtype="string")
    sample["annotated_image_path"] = pd.Series(pd.NA, index=sample.index, dtype="string")
    for field in ("width", "height", *COUNT_FIELDS):
        sample[field] = pd.Series(pd.NA, index=sample.index, dtype="Int64")
    for field in ("runtime_seconds", "inference_seconds"):
        sample[field] = pd.Series(pd.NA, index=sample.index, dtype="Float64")
    sample["timestamp"] = sample["timestamp"].astype("datetime64[us]")
    sample["camera_id"] = sample["camera_id"].astype("string")
    annotation_dir = output_dir / "annotated_images"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths: dict[int, Path] = {}
    for index, row in sample.iterrows():
        try:
            resolved_paths[index] = resolve_image_path(root, row["image_path"])
        except FileNotFoundError as exc:
            sample.loc[index, ["status", "error_message"]] = ["missing_file", str(exc)]
        except (OSError, ValueError) as exc:
            sample.loc[index, ["status", "error_message"]] = ["invalid_path", str(exc)]

    model_error = None
    model_load_seconds = None
    weights = weights_path or Path("models/yolov8n.pt")
    weights = (root / weights).resolve()
    if not sampling_only and predictor is None:
        model_started = time.perf_counter()
        try:
            if download_weights and not weights.is_file():
                if not weights.is_relative_to((root / "models").resolve()):
                    raise ValueError("Downloads must be stored under project models/")
                LOG.info("Downloading official YOLOv8 nano weights to %s", weights)
                download_default_weights(weights)
            predictor = YOLOVehicleDetector(
                weights, confidence, image_size, device, iou, max_detections
            )
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            LOG.error("Model could not load: %s", model_error)
        model_load_seconds = time.perf_counter() - model_started
    records: list[dict[str, object]] = []
    for index, path in resolved_paths.items():
        if sampling_only:
            sample.loc[index, ["status", "error_message"]] = [
                "skipped", "Sampling-only mode; image not decoded or inferred"
            ]
            continue
        if model_error:
            sample.loc[index, ["status", "error_message"]] = ["inference_error", model_error]
            continue
        image_started = time.perf_counter()
        image = None
        stage = "unreadable"
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
            sample.loc[index, ["width", "height"]] = image.size
            stage = "inference_error"
            inference_started = time.perf_counter()
            try:
                assert predictor is not None
                predictions = predictor.predict(image)
            finally:
                sample.loc[index, "inference_seconds"] = (
                    time.perf_counter() - inference_started
                )
            image_records = []
            for detection_index, prediction in enumerate(predictions):
                prediction.validate(*image.size)
                if prediction.confidence < confidence:
                    raise ValueError("Predictor returned a detection below confidence threshold")
                image_records.append({
                    "schema_version": SCHEMA_VERSION,
                    "model_run_id": sample.loc[index, "model_run_id"],
                    "image_id": sample.loc[index, "image_id"],
                    "image_path": sample.loc[index, "image_path"],
                    "camera_id": sample.loc[index, "camera_id"],
                    "timestamp": sample.loc[index, "timestamp"],
                    "detection_index": detection_index,
                    "class_id": prediction.class_id,
                    "vehicle_class": prediction.vehicle_class,
                    "confidence": prediction.confidence,
                    "x_min": prediction.x_min, "y_min": prediction.y_min,
                    "x_max": prediction.x_max, "y_max": prediction.y_max,
                })
            # Commit this image's rows only after every prediction validates.
            records.extend(image_records)
            counts = {
                f"{label}_count": sum(p.class_id == class_id for p in predictions)
                for class_id, label in COCO_VEHICLES.items()
            }
            counts["total_vehicle_count"] = sum(counts.values())
            for field, value in counts.items():
                sample.loc[index, field] = value
            sample.loc[index, "status"] = "success"
            try:
                draw = ImageDraw.Draw(image)
                for prediction in predictions:
                    draw.rectangle(
                        [prediction.x_min, prediction.y_min,
                         prediction.x_max, prediction.y_max],
                        outline="lime", width=2,
                    )
                    draw.text((prediction.x_min, prediction.y_min),
                              f"{prediction.vehicle_class} {prediction.confidence:.2f}",
                              fill="yellow", stroke_width=1, stroke_fill="black")
                annotated = annotation_dir / f"{sample.loc[index, 'image_id']}.jpg"
                image.save(annotated, format="JPEG", quality=90)
                sample.loc[index, "annotated_image_path"] = annotated.relative_to(root).as_posix()
            except Exception as exc:
                sample.loc[index, "annotation_error"] = f"{type(exc).__name__}: {exc}"
                LOG.warning("Annotation failed for %s: %s", path, exc)
        except Exception as exc:
            sample.loc[index, ["status", "error_message"]] = [
                stage, f"{type(exc).__name__}: {exc}"
            ]
            LOG.warning("%s failed for %s: %s", stage, path, exc)
        finally:
            if image is not None:
                image.close()
            sample.loc[index, "runtime_seconds"] = time.perf_counter() - image_started
        LOG.info("Image %d/%d: %s (%s)", index + 1, len(sample),
                 sample.loc[index, "image_id"][:12], sample.loc[index, "status"])

    detections = detection_frame(records)
    validate_outputs(sample, detections)
    successful = sample["status"].eq("success")
    failures = ~sample["status"].isin(["success", "skipped"])
    annotation_failures = sample["annotation_error"].notna()
    class_counts = {
        name: int(detections["vehicle_class"].eq(name).sum())
        for name in COCO_VEHICLES.values()
    }
    versions = {}
    for package in ("pandas", "pyarrow", "Pillow", "ultralytics", "torch", "torchvision"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    def runtime_stats(field: str) -> dict[str, object]:
        return json.loads(sample[field].dropna().astype(float).describe().to_json())
    weight_hash = None
    if not sampling_only and weights.is_file():
        with weights.open("rb") as stream:
            weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_run_id": sample.loc[0, "model_run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "sampling_only" if sampling_only and not failures.any()
            else "failed" if not sampling_only and not successful.any()
            else "partial_failure" if failures.any() or annotation_failures.any()
            else "complete"
        ),
        "inference_executed": bool(sample["inference_seconds"].notna().any()),
        "model_identifier": weights.name if not sampling_only else None,
        "weights_path": (
            weights.relative_to(root).as_posix()
            if weights.is_relative_to(root) else str(weights)
        ),
        "weights_sha256": weight_hash,
        "model_error": model_error,
        "parameters": {
            "sample_size": sample_size, "seed": seed, "confidence": confidence,
            "image_size": image_size, "device": device, "iou": iou,
            "max_detections": max_detections, "sampling_only": sampling_only,
            "download_weights": download_weights,
        },
        "source_manifest": "data/interim/image_manifest.parquet",
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "population_images": len(manifest),
        "eligible_strata": int(manifest.assign(
            date=manifest["timestamp"].dt.strftime("%Y-%m-%d").fillna("unknown"),
            band=pd.cut(manifest["timestamp"].dt.hour, [-1, 10, 15, 23],
                        labels=TIME_BANDS).astype("string").fillna("unknown")
        ).groupby(["camera_id", "date", "band"], observed=True).ngroups),
        "sample_coverage": _coverage(sample),
        "status_counts": {str(k): int(v) for k, v in sample["status"].value_counts().items()},
        "successful_images": int(successful.sum()),
        "zero_detection_images": int((successful & sample["total_vehicle_count"].eq(0)).sum()),
        "failed_images": int(failures.sum()),
        "annotation_errors": int(annotation_failures.sum()),
        "annotated_images": int(sample["annotated_image_path"].notna().sum()),
        "detection_count": len(detections),
        "per_class_detection_counts": class_counts,
        "errors": sample.loc[failures, ["image_id", "image_path", "status", "error_message"]]
            .to_dict(orient="records"),
        "annotation_error_details": sample.loc[
            annotation_failures, ["image_id", "image_path", "annotation_error"]
        ].to_dict(orient="records"),
        "runtime": {
            "units": "seconds", "model_load_seconds": model_load_seconds,
            "image_runtime_statistics": runtime_stats("runtime_seconds"),
            "inference_runtime_statistics": runtime_stats("inference_seconds"),
            "elapsed_before_output_write_seconds": time.perf_counter() - started,
            "first_call_warmup_included": True,
        },
        "package_versions": versions,
        "timestamp_timezone": "naive; unchanged filename clock time",
        "limitations": LIMITATIONS,
        "evaluation_metrics": None,
        "evaluation_note": "No annotated ground truth; precision/recall/accuracy not measured.",
    }
    sample.to_parquet(output_dir / "sample_manifest.parquet", index=False)
    detections.to_parquet(output_dir / "detections.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOG.info("Saved feasibility artifacts to %s", output_dir)
    return summary
