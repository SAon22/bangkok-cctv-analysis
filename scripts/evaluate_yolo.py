"""Evaluate saved baseline/low-confidence predictions in reviewed full/ROI scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_evaluation_scope import validate_scope  # noqa: E402
from bangkok_cctv.engineering.detection_evaluation import (  # noqa: E402
    CLASS_NAMES, COCO_TO_GT, Box, center_in_scope, match_boxes, metrics, read_labels,
)
from bangkok_cctv.engineering.yolo_feasibility import resolve_image_path  # noqa: E402

CONFIGURATIONS = ("baseline", "low_confidence")


def evaluate(root: Path, output: Path) -> dict:
    """Validate all inputs, calculate deterministic matches, then save new outputs."""
    root = root.resolve()
    base = root / "data/interim/yolo_ground_truth"
    diagnostic = root / "data/interim/yolo_diagnostics/step45a_02"
    output = (root / output).resolve()
    allowed = (base / "evaluation").resolve()
    if not output.is_relative_to(allowed):
        raise ValueError("Outputs must be under yolo_ground_truth/evaluation")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output-dir")
    names = {p.name for p in (base / "images").iterdir()
             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if len(names) != 8:
        raise ValueError("Expected exactly eight annotation images")
    validate_scope(base / "evaluation_scope.csv", names, require_reviewed=True)
    classes = (base / "labels/classes.txt").read_text(encoding="utf-8-sig").strip().splitlines()
    if tuple(value.strip() for value in classes) != CLASS_NAMES:
        raise ValueError("classes.txt must define car,motorcycle,bus,truck in that order")
    labels = {p.stem for p in (base / "labels").glob("*.txt") if p.name != "classes.txt"}
    if labels != {Path(name).stem for name in names}:
        raise ValueError("Expected one label file per image with no extra labels")
    with (base / "image_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
        mapping = [{key.strip(): value.strip() for key, value in row.items()}
                   for row in csv.DictReader(stream)]
    if (len(mapping) != 8 or {row["copied_filename"] for row in mapping} != names
            or len({row["image_path"] for row in mapping}) != 8):
        raise ValueError("Mapping must join exactly eight unique originals and copies")
    with (base / "evaluation_scope.csv").open(encoding="utf-8-sig", newline="") as stream:
        scopes = {row["image_filename"].strip(): row for row in csv.DictReader(stream)}
    provenance = json.loads((diagnostic / "provenance.json").read_text(encoding="utf-8"))
    detections = pd.read_csv(diagnostic / "detections.csv", dtype={"camera_id": "string"})
    statuses = pd.read_csv(diagnostic / "image_results.csv", dtype={"camera_id": "string"})
    required = {"image_path", "configuration", "class_id", "class_name", "confidence",
                "x_min", "y_min", "x_max", "y_max"}
    if not required <= set(detections.columns):
        raise ValueError(f"Detection CSV missing columns: {sorted(required-set(detections))}")
    if not {"image_path", "configuration", "status", "detection_count", "width", "height"} <= set(statuses):
        raise ValueError("Configuration-status CSV is missing required columns")
    originals = {row["image_path"] for row in mapping}
    selected = detections.loc[detections.configuration.isin(CONFIGURATIONS)].copy()
    status = statuses.loc[statuses.configuration.isin(CONFIGURATIONS)]
    if not set(selected.image_path) <= originals:
        raise ValueError("Predictions reference unknown evaluation images")
    for configuration in CONFIGURATIONS:
        coverage = status.loc[status.configuration.eq(configuration)]
        if (len(coverage) != 8 or set(coverage.image_path) != originals
                or not coverage.status.eq("success").all()):
            raise ValueError(f"Incomplete/failed prediction coverage: {configuration}")
    selected["prediction_csv_row"] = selected.index + 2
    prepared, hashes = [], {}
    input_files = [base / "image_mapping.csv", base / "evaluation_scope.csv",
                   base / "labels/classes.txt", diagnostic / "detections.csv",
                   diagnostic / "image_results.csv", diagnostic / "provenance.json"]
    for row in sorted(mapping, key=lambda item: item["copied_filename"]):
        filename, original_path = row["copied_filename"], row["image_path"]
        if Path(filename).name != filename:
            raise ValueError("Copied filename must be a basename")
        source = resolve_image_path(root, original_path)
        copied = base / "images" / filename
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
        if (digest != hashlib.sha256(source.read_bytes()).hexdigest()
                or digest != provenance["image_sha256"].get(original_path)):
            raise ValueError(f"Image/provenance checksum mismatch: {filename}")
        with Image.open(copied) as image:
            width, height = image.size
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"EXIF-oriented labels require explicit handling: {filename}")
            image.load()
        label = base / "labels" / f"{copied.stem}.txt"
        ground_truth = read_labels(label, width, height)
        scope = scopes[filename]
        bounds = tuple(float(scope[key]) for key in ["x_min", "y_min", "x_max", "y_max"])
        included_gt = [i for i, box in enumerate(ground_truth)
                       if center_in_scope(box, bounds, width, height)]
        prepared.append((row, scope["scope"].strip().lower(), bounds, width, height,
                         ground_truth, included_gt))
        input_files.extend([copied, source, label])
    for path in input_files:
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    per_class, per_image, match_rows = [], [], []
    for row, scope, bounds, width, height, gt, gt_ids in prepared:
        identity = {"image_filename": row["copied_filename"], "image_path": row["image_path"],
                    "scope": scope, **dict(zip(["x_min", "y_min", "x_max", "y_max"], bounds))}
        truth = [gt[i] for i in gt_ids]
        for configuration in CONFIGURATIONS:
            part = selected.loc[selected.image_path.eq(row["image_path"])
                                & selected.configuration.eq(configuration)]
            saved = status.loc[status.image_path.eq(row["image_path"])
                               & status.configuration.eq(configuration)].iloc[0]
            if (saved.detection_count != len(part) or saved.width != width
                    or saved.height != height):
                raise ValueError(f"Prediction count/dimension mismatch: {row['copied_filename']}")
            predictions, prediction_ids = [], []
            for detection in part.itertuples(index=False):
                class_id = int(detection.class_id)
                if (class_id != detection.class_id or class_id not in COCO_TO_GT
                        or detection.class_name != CLASS_NAMES[COCO_TO_GT[class_id]]):
                    raise ValueError("Unexpected COCO class ID/name mapping")
                coordinates = tuple(float(getattr(detection, key))
                                    for key in ["x_min", "y_min", "x_max", "y_max"])
                x1, y1, x2, y2 = coordinates
                threshold = .25 if configuration == "baseline" else .05
                if not (all(math.isfinite(v) for v in coordinates)
                        and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                        and math.isfinite(detection.confidence)
                        and threshold <= detection.confidence <= 1):
                    raise ValueError("Invalid prediction coordinates/confidence")
                box = Box(COCO_TO_GT[class_id], coordinates)
                if center_in_scope(box, bounds, width, height):
                    predictions.append(box)
                    prediction_ids.append(int(detection.prediction_csv_row))
            matches = match_boxes(truth, predictions, .5)
            matched_gt = {item[0] for item in matches}
            matched_predictions = {item[1] for item in matches}
            for gi, pi, overlap in matches:
                match_rows.append({
                    **identity, "configuration": configuration, "outcome": "tp",
                    "class_name": CLASS_NAMES[truth[gi].class_id],
                    "gt_box_index": gt_ids[gi], "prediction_csv_row": prediction_ids[pi],
                    "iou": overlap,
                })
            for gi, box in enumerate(truth):
                if gi not in matched_gt:
                    match_rows.append({**identity, "configuration": configuration, "outcome": "fn",
                                       "class_name": CLASS_NAMES[box.class_id],
                                       "gt_box_index": gt_ids[gi]})
            for pi, box in enumerate(predictions):
                if pi not in matched_predictions:
                    match_rows.append({**identity, "configuration": configuration, "outcome": "fp",
                                       "class_name": CLASS_NAMES[box.class_id],
                                       "prediction_csv_row": prediction_ids[pi]})
            for class_id, name in enumerate(CLASS_NAMES):
                per_class.append({
                    **identity, "configuration": configuration, "class_name": name,
                    **metrics(sum(box.class_id == class_id for box in truth),
                              sum(box.class_id == class_id for box in predictions),
                              sum(truth[gi].class_id == class_id for gi, _, _ in matches)),
                })
            error = len(predictions) - len(truth)
            per_image.append({
                **identity, "configuration": configuration,
                "gt_boxes_before_scope": len(gt), "gt_count": len(truth),
                "predictions_before_scope": len(part), "predicted_count": len(predictions),
                "signed_count_error": error, "absolute_count_error": abs(error),
                **{key: value for key, value in metrics(len(truth), len(predictions),
                                                        len(matches)).items()
                   if key not in {"gt_count", "predicted_count"}},
            })
    class_frame, image_frame = pd.DataFrame(per_class), pd.DataFrame(per_image)
    aggregates = []
    for configuration in CONFIGURATIONS:
        group = class_frame.loc[class_frame.configuration.eq(configuration)]
        for name in [*CLASS_NAMES, "micro"]:
            part = group if name == "micro" else group.loc[group.class_name.eq(name)]
            aggregates.append({
                "configuration": configuration, "class_name": name,
                **metrics(int(part.gt_count.sum()), int(part.predicted_count.sum()),
                          int(part.tp.sum())),
            })
    report = {
        "schema_version": "roi-evaluation-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": 8, "scope_counts": pd.Series([item[1] for item in prepared]).value_counts().to_dict(),
        "iou_threshold": .5,
        "matching": "Descending IoU greedy; ties: GT index then prediction index; class-aware",
        "roi_policy": "Inclusive normalized box centers; unchanged original boxes for IoU; no clipping",
        "undefined_metrics": "null; F1=2TP/(2TP+FP+FN), null only if denominator is zero",
        "class_mapping_gt": dict(enumerate(CLASS_NAMES)), "class_mapping_coco_to_gt": COCO_TO_GT,
        "metrics": aggregates,
        "count_mae": {configuration: float(image_frame.loc[
            image_frame.configuration.eq(configuration), "absolute_count_error"].mean())
            for configuration in CONFIGURATIONS},
        "input_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(
            (ROOT / "src/bangkok_cctv/engineering/detection_evaluation.py").read_bytes()
        ).hexdigest(),
        "limitations": [
            "Eight hand-picked diagnostic images, not a representative holdout.",
            "Mixed full/ROI scopes; results describe reviewed regions only.",
            "Completeness and class choices rely on manual review; no uncertainty intervals.",
            "Review notes still contain preparation-era pending text; reviewed CSV is authoritative.",
            "No inference rerun, training, AP/mAP, or traffic-flow interpretation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(aggregates).to_csv(output / "metrics.csv", index=False)
    image_frame.to_csv(output / "per_image.csv", index=False)
    class_frame.to_csv(output / "per_image_class.csv", index=False)
    pd.DataFrame(match_rows, columns=[
        "image_filename", "image_path", "scope", "x_min", "y_min", "x_max", "y_max",
        "configuration", "outcome", "class_name", "gt_box_index", "prediction_csv_row", "iou",
    ]).to_csv(output / "matches.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/interim/yolo_ground_truth/evaluation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = evaluate(args.project_root, args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logging.error("Evaluation stopped: %s", exc)
        return 1
    print(pd.DataFrame(report["metrics"]).to_string(index=False))
    print("Count MAE:", report["count_mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
