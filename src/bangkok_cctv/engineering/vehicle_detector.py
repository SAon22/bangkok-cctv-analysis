"""Reusable local-weight YOLO vehicle inference; imports stay lightweight."""

from __future__ import annotations

import math
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
)


@dataclass(frozen=True)
class VehicleDetection:
    """One predicted COCO vehicle with pixel xyxy coordinates, not ground truth."""

    class_id: int
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def vehicle_class(self) -> str:
        return COCO_VEHICLES[self.class_id]

    def validate(self, width: int, height: int) -> None:
        values = (self.confidence, self.x_min, self.y_min, self.x_max, self.y_max)
        if self.class_id not in COCO_VEHICLES:
            raise ValueError(f"Unsupported COCO vehicle class: {self.class_id}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detection contains non-finite coordinates/confidence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        if not (0 <= self.x_min < self.x_max <= width
                and 0 <= self.y_min < self.y_max <= height):
            raise ValueError("Detection box is outside original-image bounds")


class VehiclePredictor(Protocol):
    """Small mockable interface used by feasibility orchestration."""

    def predict(self, image: Image.Image) -> list[VehicleDetection]: ...


def download_default_weights(destination: Path) -> Path:
    """Explicit opt-in download of official YOLOv8 nano weights; never overwrite."""
    destination = destination.resolve()
    if destination.is_file():
        return destination
    if destination.name != "yolov8n.pt":
        raise ValueError("--download-weights only supports the default yolov8n.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.download")
    if temporary.exists():
        raise FileExistsError(f"Existing partial download: {temporary}")
    try:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as source:
            with temporary.open("xb") as target:
                shutil.copyfileobj(source, target)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


class YOLOVehicleDetector:
    """Load once and predict repeatedly using an Ultralytics COCO detector."""

    def __init__(
        self, weights_path: Path, confidence: float = 0.25,
        image_size: int = 640, device: str = "cpu", iou: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not math.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError("iou must be between 0 and 1")
        if image_size < 32 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if max_detections < 1:
            raise ValueError("max_detections must be positive")
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. Supply local weights "
                "or opt in to --download-weights."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics/PyTorch unavailable. Install the declared ultralytics "
                "dependency in the selected Python environment."
            ) from exc
        self.model = YOLO(str(weights_path.resolve()), task="detect")
        if self.model.task != "detect":
            raise ValueError("Weights must be an object detection model")
        if any(self.model.names.get(key) != value
               for key, value in COCO_VEHICLES.items()):
            raise ValueError("Weights do not expose the required COCO class mapping")
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.iou = iou
        self.max_detections = max_detections

    def predict(self, image: Image.Image) -> list[VehicleDetection]:
        """Predict from an in-memory PIL image, avoiding Unicode-path CV readers."""
        results = self.model.predict(
            source=image, conf=self.confidence, classes=list(COCO_VEHICLES),
            imgsz=self.image_size, device=self.device, iou=self.iou,
            max_det=self.max_detections, save=False, save_txt=False,
            verbose=False, stream=False, augment=False,
        )
        if len(results) != 1 or results[0].boxes is None:
            raise RuntimeError("YOLO returned no usable single-image detection result")
        boxes = results[0].boxes
        predictions = []
        for coordinates, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(), strict=True,
        ):
            class_id = int(class_id)
            if class_id not in COCO_VEHICLES or confidence < self.confidence:
                continue
            detection = VehicleDetection(class_id, float(confidence), *map(float, coordinates))
            detection.validate(*image.size)
            predictions.append(detection)
        return predictions


def detect_vehicles(image_path: Path, weights_path: Path) -> dict[str, int]:
    """Preserve the original API: return four class counts and their total.

    For repeated calls use YOLOVehicleDetector to avoid loading weights per image.
    Read/inference errors propagate; they must never become zero-vehicle results.
    """
    detector = YOLOVehicleDetector(weights_path)
    with Image.open(image_path) as source:
        with source.convert("RGB") as image:
            predictions = detector.predict(image)
    counts = {f"{name}_count": 0 for name in COCO_VEHICLES.values()}
    for detection in predictions:
        counts[f"{detection.vehicle_class}_count"] += 1
    counts["total_vehicle_count"] = sum(counts.values())
    return counts
