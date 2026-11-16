"""Validate the finalized Step 5 run without inference or source modification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangkok_cctv.engineering.detection_validation import validate_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("data/interim/detection_runs/yolov8n_baseline_full_01"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/interim/image_manifest.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/interim/detection_validation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = validate_run(args.project_root, args.run_dir, args.manifest, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.error("Validation stopped: %s", exc)
        return 1
    print(json.dumps({key: report[key] for key in
                      ("status", "statistics", "issue_counts", "readiness_recommendation",
                       "validation_seconds")}, indent=2))
    return 0 if report["issue_counts"]["critical"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
