"""Sample CCTV images and optionally run experimental YOLO vehicle inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangkok_cctv.engineering.yolo_feasibility import run_feasibility  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/interim/yolo_feasibility"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-only", "--dry-run", action="store_true",
                        help="Save a sample and validate paths; no model/image decode")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--download-weights", action="store_true",
                        help="Explicitly download official yolov8n.pt into models/")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = run_feasibility(
            args.project_root, args.output_dir, sample_size=args.sample_size,
            seed=args.seed, sampling_only=args.sampling_only, weights_path=args.weights,
            confidence=args.confidence, image_size=args.image_size,
            device=args.device, iou=args.iou, max_detections=args.max_detections,
            download_weights=args.download_weights,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        logging.error("Feasibility run failed: %s", exc)
        return 1
    print(json.dumps({
        field: summary[field] for field in [
            "status", "inference_executed", "sample_coverage", "successful_images",
            "zero_detection_images", "failed_images", "annotation_errors",
            "detection_count", "per_class_detection_counts", "model_error",
        ]
    }, indent=2))
    return int(summary["status"] in {"failed", "partial_failure"})


if __name__ == "__main__":
    raise SystemExit(main())
