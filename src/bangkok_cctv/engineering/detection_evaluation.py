"""Pure helpers for diagnostic ground-truth evaluation; no model inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ("car", "motorcycle", "bus", "truck")
COCO_TO_GT = {2: 0, 3: 1, 5: 2, 7: 3}


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]


def read_labels(path: Path, width: int, height: int) -> list[Box]:
    """Parse YOLO labels, allowing 1e-6 normalized edge-rounding tolerance."""
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("expected five fields")
            class_id = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if class_id not in range(4):
                raise ValueError("class ID must be 0-3")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
                raise ValueError("normalized coordinates must be finite and in [0,1]")
            if w <= 0 or h <= 0:
                raise ValueError("width and height must be positive")
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:
                raise ValueError("box extends outside the image")
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
        boxes.append(Box(class_id, (x1 * width, y1 * height,
                                    x2 * width, y2 * height)))
    return boxes


def center_in_scope(
    box: Box, bounds: tuple[float, float, float, float], width: int, height: int,
) -> bool:
    """Use inclusive normalized center bounds; do not clip the box."""
    x1, y1, x2, y2 = box.xyxy
    cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
    return bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]


def iou(first: Box, second: Box) -> float:
    """Continuous pixel-coordinate IoU, without a '+1 pixel' adjustment."""
    a, b = first.xyxy, second.xyxy
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def match_boxes(
    truth: list[Box], predictions: list[Box], threshold: float = .5,
) -> list[tuple[int, int, float]]:
    """Greedy descending-IoU, then GT/prediction index; each box matches once."""
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be in [0,1]")
    candidates = []
    for gi, gt in enumerate(truth):
        for pi, prediction in enumerate(predictions):
            if gt.class_id == prediction.class_id:
                overlap = iou(gt, prediction)
                if overlap >= threshold:
                    candidates.append((-overlap, gi, pi))
    matches = []
    used_gt, used_predictions = set(), set()
    for negative_iou, gi, pi in sorted(candidates):
        if gi not in used_gt and pi not in used_predictions:
            matches.append((gi, pi, -negative_iou))
            used_gt.add(gi)
            used_predictions.add(pi)
    return matches


def metrics(gt_count: int, prediction_count: int, tp: int) -> dict:
    """Undefined ratios are None; count-based F1 is zero when errors exist."""
    fp, fn = prediction_count - tp, gt_count - tp
    return {
        "gt_count": gt_count, "predicted_count": prediction_count,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / prediction_count if prediction_count else None,
        "recall": tp / gt_count if gt_count else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
