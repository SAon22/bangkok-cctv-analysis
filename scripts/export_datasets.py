"""Export the validated Step 5 run as versioned consumer Parquet tables."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangkok_cctv.engineering.export_datasets import export_datasets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, default=Path(
        "data/interim/detection_runs/yolov8n_baseline_full_01"))
    parser.add_argument("--manifest-path", type=Path,
                        default=Path("data/interim/image_manifest.parquet"))
    parser.add_argument("--camera-csv", type=Path, default=Path("data/raw/cctv_info.csv"))
    parser.add_argument("--validation-report", type=Path, default=Path(
        "data/interim/detection_validation/data_quality_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = export_datasets(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.error("Export stopped: %s", exc)
        return 1
    print(json.dumps({"contract_version": result["contract_version"],
                      "files": result["files"],
                      "zero_detection_images": result["successful_zero_detection_images"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
