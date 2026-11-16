"""Compare post-NMS YOLO predictions for an explicit list of 1-8 images.

Default: validate the list only. Use --run only after the user confirms the list.
No sampling, weight download, pre-NMS inspection, or accuracy claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangkok_cctv.engineering.vehicle_detector import (  # noqa: E402
    COCO_VEHICLES, YOLOVehicleDetector,
)
from bangkok_cctv.engineering.yolo_feasibility import resolve_image_path  # noqa: E402

CONFIGS = [
    {"name": "baseline", "confidence": .25, "classes": [2, 3, 5, 7], "iou": .7},
    {"name": "low_confidence", "confidence": .05, "classes": [2, 3, 5, 7], "iou": .7},
    {"name": "all_classes", "confidence": .05, "classes": None, "iou": .7},
    {"name": "nms_diagnostic", "confidence": .05, "classes": [2, 3, 5, 7], "iou": .95},
]
COLUMNS = [
    "image_path", "camera_id", "timestamp", "configuration", "class_id",
    "class_name", "confidence", "x_min", "y_min", "x_max", "y_max",
]


def load_images(root: Path, list_path: Path) -> list[dict[str, Any]]:
    """Require explicit unique manifest paths; never substitute missing images."""
    document = json.loads((root / list_path).read_text(encoding="utf-8"))
    entries = document.get("images") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError("JSON must contain an images list with 1-8 entries")
    manifest = pd.read_parquet(root / "data/interim/image_manifest.parquet",
                               columns=["image_path", "camera_id", "timestamp"])
    if manifest["image_path"].duplicated().any():
        raise ValueError("Manifest image_path values must be unique")
    manifest = manifest.set_index("image_path")
    selected = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"image_path", "note", "nms"}:
            raise ValueError("Each entry supports only image_path, note and nms")
        path = entry.get("image_path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Each image_path must be a unique nonempty string")
        if not isinstance(entry.get("note", ""), str):
            raise ValueError(f"note must be a string: {path}")
        if not isinstance(entry.get("nms", False), bool):
            raise ValueError(f"nms must be true or false: {path}")
        if path not in manifest.index:
            raise ValueError(f"Selected path is absent from the manifest: {path}")
        resolve_image_path(root, path)
        row = manifest.loc[path]
        selected.append({
            "image_path": path, "note": entry.get("note", ""),
            "nms": entry.get("nms", False), "camera_id": str(row.camera_id),
            "timestamp": None if pd.isna(row.timestamp) else row.timestamp.isoformat(),
        })
        seen.add(path)
    return selected


def panel(image: Image.Image, predictions: list[dict], title: str) -> Image.Image:
    """Keep source pixels at 1:1 scale; add a title above the annotation."""
    canvas = Image.new("RGB", (image.width, image.height + 45), "white")
    canvas.paste(image, (0, 45))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill="black")
    for row in predictions:
        box = (row["x_min"], row["y_min"] + 45,
               row["x_max"], row["y_max"] + 45)
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]),
                  f"{row['class_name']} {row['confidence']:.2f}",
                  fill="yellow", stroke_width=1, stroke_fill="black")
    return canvas


def run_diagnostics(
    root: Path, list_path: Path, output: Path, weights: Path,
) -> dict[str, Any]:
    """Execute only after confirmation; save a separate, non-overwriting run."""
    root = root.resolve()
    selected = load_images(root, list_path)
    output = (root / output).resolve()
    base = (root / "data/interim/yolo_diagnostics").resolve()
    if output == base or not output.is_relative_to(base):
        raise ValueError("Choose a NEW run subdirectory under data/interim/yolo_diagnostics/")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    weights = (root / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trusted local weights required: {weights}; no downloads")
    previous = json.loads(
        (root / "data/interim/yolo_feasibility/summary.json").read_text(encoding="utf-8")
    )
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weight_hash != previous.get("weights_sha256"):
        raise ValueError("Weights differ from the existing feasibility run")
    # Reuse load/task/COCO checks; bypass vehicle-only predict() for all-class results.
    model = YOLOVehicleDetector(weights, image_size=640, device="cpu").model
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, results = [], []
    provenance = {
        "schema_version": "diagnostics-0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_list": selected, "configurations": CONFIGS,
        "image_size": 640, "device": "cpu", "max_detections": 300,
        "agnostic_nms": False, "rect": True, "augment": False,
        "weights": weights.name, "weights_sha256": weight_hash,
        "manifest_sha256": hashlib.sha256(
            (root / "data/interim/image_manifest.parquet").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "versions": {name: version(name) for name in
                     ["ultralytics", "torch", "torchvision", "Pillow", "pandas"]},
        "coordinate_frame": "EXIF-oriented original decoded pixels; no manual resizing",
        "interpretation": "Post-NMS model predictions only; no TP/FP/FN or accuracy claims",
    }
    for index, entry in enumerate(selected, 1):
        configs = CONFIGS if entry["nms"] else CONFIGS[:3]
        identity = {key: entry[key] for key in ["image_path", "camera_id", "timestamp"]}
        image = None
        try:
            with Image.open(resolve_image_path(root, entry["image_path"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:
            for config in configs:
                results.append({**identity, "configuration": config["name"],
                                "status": "unreadable", "detection_count": None,
                                "error": str(exc)})
            continue
        panels = [panel(image, [], "Original (unscaled)")]
        image_hash = hashlib.sha256(entry["image_path"].encode("utf-8")).hexdigest()
        provenance.setdefault("image_sha256", {})[entry["image_path"]] = hashlib.sha256(
            resolve_image_path(root, entry["image_path"]).read_bytes()
        ).hexdigest()
        try:
            for config in configs:
                tick = time.perf_counter()
                rows = []
                error = None
                try:
                    prediction = model.predict(
                        source=image, conf=config["confidence"], classes=config["classes"],
                        iou=config["iou"], imgsz=640, device="cpu", max_det=300,
                        agnostic_nms=False, rect=True, augment=False,
                        save=False, save_txt=False, verbose=False, stream=False,
                    )[0]
                    if tuple(prediction.orig_shape) != (image.height, image.width):
                        raise ValueError("Prediction coordinates do not match the source image")
                    boxes = prediction.boxes
                    for xyxy, confidence, class_id in zip(
                        boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                        boxes.cls.cpu().tolist(), strict=True,
                    ):
                        x1, y1, x2, y2 = map(float, xyxy)
                        if not (all(math.isfinite(v) for v in [*xyxy, confidence])
                                and 0 <= x1 < x2 <= image.width
                                and 0 <= y1 < y2 <= image.height
                                and config["confidence"] <= confidence <= 1):
                            raise ValueError("Invalid confidence or source-coordinate box")
                        class_id = int(class_id)
                        if config["classes"] is not None and class_id not in COCO_VEHICLES:
                            raise ValueError("Unexpected class in vehicle-filtered result")
                        rows.append({
                            **identity, "configuration": config["name"],
                            "class_id": class_id, "class_name": prediction.names[class_id],
                            "confidence": float(confidence), "x_min": x1,
                            "y_min": y1, "x_max": x2, "y_max": y2,
                        })
                    records.extend(rows)
                except Exception as exc:
                    rows = []
                    error = f"{type(exc).__name__}: {exc}"
                    logging.warning("%s / %s: %s", entry["image_path"], config["name"], error)
                results.append({
                    **identity, "configuration": config["name"],
                    "status": "error" if error else "success",
                    "detection_count": None if error else len(rows), "error": error,
                    "runtime_seconds": time.perf_counter() - tick,
                    "width": image.width, "height": image.height,
                })
                title = (f"{config['name']} | {'ERROR' if error else str(len(rows)) + ' predictions'}"
                         f"\nconf={config['confidence']} iou={config['iou']}")
                panels.append(panel(image, rows, title))
            comparison = Image.new("RGB", (image.width * len(panels), image.height + 45))
            for column, rendered in enumerate(panels):
                comparison.paste(rendered, (column * image.width, 0))
            name = f"{index:02d}_camera_{image_hash[:12]}_comparison.png"
            comparison.save(output / name)
            comparison.close()
            entry["comparison_file"] = name
        finally:
            image.close()
            for rendered in panels:
                rendered.close()
    pd.DataFrame(records, columns=COLUMNS).to_csv(output / "detections.csv", index=False)
    pd.DataFrame(results).to_csv(output / "image_results.csv", index=False)
    provenance.update(
        elapsed_seconds=time.perf_counter() - started,
        failed_configurations=sum(row["status"] != "success" for row in results),
        detection_rows=len(records),
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Explicit JSON image list")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--run", action="store_true",
                        help="Run inference ONLY after user confirmation; default is validation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if not args.run:
            print(json.dumps(load_images(ROOT, args.images), indent=2, ensure_ascii=False))
            print("Validation only. No inference. Confirm the exact list before using --run.")
            return 0
        if args.output_dir is None:
            raise ValueError("--run requires a NEW --output-dir")
        report = run_diagnostics(ROOT, args.images, args.output_dir, args.weights)
        print(f"Saved {report['detection_rows']} predictions; "
              f"{report['failed_configurations']} failed configurations")
        return int(report["failed_configurations"] > 0)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
