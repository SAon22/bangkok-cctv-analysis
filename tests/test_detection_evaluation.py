"""Focused matching, annotation, scope and saved-prediction evaluation tests."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bangkok_cctv.engineering.detection_evaluation import (
    Box, center_in_scope, iou, match_boxes, metrics, read_labels,
)


def test_perfect_match_and_iou_threshold() -> None:
    gt = [Box(0, (0, 0, 10, 10))]
    assert match_boxes(gt, gt) == [(0, 0, 1.0)]
    half = Box(0, (0, 0, 5, 10))
    assert iou(gt[0], half) == .5
    assert match_boxes(gt, [half]) == [(0, 0, .5)]
    assert metrics(1, 1, 1)["f1"] == 1


def test_duplicate_prediction_counts_as_fp_and_ties_are_stable() -> None:
    box = Box(0, (0, 0, 10, 10))
    matches = match_boxes([box], [box, box])
    assert matches == [(0, 0, 1.0)]
    result = metrics(1, 2, len(matches))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["precision"] == .5


def test_class_mismatch_creates_fp_and_fn() -> None:
    assert match_boxes([Box(0, (0, 0, 10, 10))],
                       [Box(1, (0, 0, 10, 10))]) == []
    assert metrics(1, 1, 0)["fp"] == metrics(1, 1, 0)["fn"] == 1


def test_zero_predictions_and_undefined_denominators() -> None:
    assert match_boxes([Box(0, (0, 0, 10, 10))], []) == []
    result = metrics(1, 0, 0)
    assert result["precision"] is None and result["recall"] == result["f1"] == 0
    empty = metrics(0, 0, 0)
    assert empty["precision"] is empty["recall"] is empty["f1"] is None
    assert metrics(0, 1, 0)["recall"] is None


@pytest.mark.parametrize("center,expected", [
    ((.25, .5), True), ((.75, .5), True), ((.5, .25), True),
    ((.5, .75), True), ((.249, .5), False), ((.5, .751), False),
])
def test_roi_center_boundary_inclusive(center: tuple, expected: bool) -> None:
    x, y = center
    box = Box(0, ((x-.05)*100, (y-.05)*100, (x+.05)*100, (y+.05)*100))
    assert center_in_scope(box, (.25, .25, .75, .75), 100, 100) == expected


def test_roi_does_not_clip_boxes_for_iou() -> None:
    large = Box(0, (0, 0, 100, 100))
    small = Box(0, (25, 25, 75, 75))
    assert center_in_scope(large, (.25, .25, .75, .75), 100, 100)
    assert match_boxes([small], [large]) == []
    assert iou(small, large) == .25


def test_greedy_uses_highest_overlap_before_input_order() -> None:
    gt = [Box(0, (0, 0, 10, 10))]
    predictions = [Box(0, (0, 0, 8, 10)), Box(0, (0, 0, 10, 10))]
    assert match_boxes(gt, predictions) == [(0, 1, 1.0)]


@pytest.mark.parametrize("text", [
    "0 .5 .5 .1", "4 .5 .5 .1 .1", "0 nan .5 .1 .1",
    "0 .5 .5 0 .1", "0 .99 .5 .2 .1", "0 -0.1 .5 .1 .1",
])
def test_invalid_annotations_fail(tmp_path: Path, text: str) -> None:
    path = tmp_path / "label.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(path, 100, 100)


def test_annotation_conversion_and_rounding_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "label.txt"
    path.write_text("1 .5 .5 .2 .4\n0 .95 .5 .100001 .1\n", encoding="utf-8")
    boxes = read_labels(path, 100, 200)
    assert boxes[0] == Box(1, (40, 60, 60, 140))
    assert boxes[1].xyxy[2] > 100  # Preserve rounding, do not clip for IoU.


def test_cli_evaluates_saved_predictions_and_refuses_overwrite(tmp_path: Path) -> None:
    base = tmp_path / "data/interim/yolo_ground_truth"
    diagnostic = tmp_path / "data/interim/yolo_diagnostics/step45a_02"
    (base / "images").mkdir(parents=True)
    (base / "labels").mkdir()
    diagnostic.mkdir(parents=True)
    (base / "labels/classes.txt").write_text("car\nmotorcycle\nbus\ntruck\n")
    mapping, scopes, predictions, statuses, hashes = [], [], [], [], {}
    for index in range(8):
        original = tmp_path / f"data/raw/{index}.png"
        original.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (100, 100)) as image:
            image.save(original)
        name = f"camera_{index:03}.png"
        copied = base / "images" / name
        copied.write_bytes(original.read_bytes())
        path = original.relative_to(tmp_path).as_posix()
        hashes[path] = hashlib.sha256(original.read_bytes()).hexdigest()
        mapping.append({"image_path": path, "copied_filename": name})
        roi = index < 3
        scopes.append([name, "roi" if roi else "full", 0, .3 if roi else 0, 1, 1, "true"])
        (base / "labels" / f"camera_{index:03}.txt").write_text("0 .5 .5 .2 .2\n")
        for configuration in ["baseline", "low_confidence"]:
            count = (2 if index == 0 else 1) if configuration == "baseline" else 0
            statuses.append({"image_path": path, "configuration": configuration,
                             "status": "success", "detection_count": count,
                             "width": 100, "height": 100})
            for _ in range(count):
                predictions.append({
                    "image_path": path, "configuration": configuration, "camera_id": f"{index:03}",
                    "class_id": 2, "class_name": "car", "confidence": .8,
                    "x_min": 40, "y_min": 40, "x_max": 60, "y_max": 60,
                })
    pd.DataFrame(mapping).to_csv(base / "image_mapping.csv", index=False)
    with (base / "evaluation_scope.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_filename", "scope", "x_min", "y_min", "x_max", "y_max", "reviewed"])
        writer.writerows(scopes)
    pd.DataFrame(predictions).to_csv(diagnostic / "detections.csv", index=False)
    pd.DataFrame(statuses).to_csv(diagnostic / "image_results.csv", index=False)
    (diagnostic / "provenance.json").write_text(json.dumps({"image_sha256": hashes}))
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_yolo.py"
    command = [sys.executable, "-B", str(script), "--project-root", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((base / "evaluation/summary.json").read_text(encoding="utf-8"))
    overall = {row["configuration"]: row for row in summary["metrics"] if row["class_name"] == "micro"}
    assert overall["baseline"]["tp"] == 8 and overall["baseline"]["fp"] == 1
    assert overall["low_confidence"]["fn"] == 8
    assert summary["count_mae"] == {"baseline": .125, "low_confidence": 1.0}
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 1 and "Refusing to overwrite" in repeated.stderr
