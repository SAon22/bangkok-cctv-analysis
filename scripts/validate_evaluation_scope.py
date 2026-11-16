"""Validate manual full-image/ROI review scope without running evaluation."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["image_filename", "scope", "x_min", "y_min", "x_max", "y_max", "reviewed"]


def validate_scope(
    path: Path, image_names: set[str], *, require_reviewed: bool = False,
) -> dict[str, int | bool]:
    """Validate draft syntax; require_reviewed additionally gates evaluation."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected CSV columns in this order: {FIELDS}")
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {line}: wrong number of fields")
    names = [row["image_filename"].strip() for row in rows]
    if len(names) != len(set(names)) or set(names) != image_names:
        raise ValueError("Scope filenames must match the images exactly, once each")
    reviewed_count = 0
    pending_count = 0
    for line, row in enumerate(rows, 2):
        scope = row["scope"].strip().lower()
        reviewed = row["reviewed"].strip().lower()
        bounds = [row[key].strip() for key in FIELDS[2:6]]
        if scope not in {"pending", "full", "roi"}:
            raise ValueError(f"CSV row {line}: scope must be pending, full or roi")
        if reviewed not in {"true", "false"}:
            raise ValueError(f"CSV row {line}: reviewed must be true or false")
        if scope == "pending":
            if reviewed != "false" or any(bounds):
                raise ValueError(f"CSV row {line}: pending requires blank bounds and false")
            pending_count += 1
        else:
            try:
                x1, y1, x2, y2 = map(float, bounds)
            except ValueError as exc:
                raise ValueError(f"CSV row {line}: four numeric bounds required") from exc
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in [x1, y1, x2, y2]):
                raise ValueError(f"CSV row {line}: bounds must be finite and within [0,1]")
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"CSV row {line}: ROI must have positive width and height")
            if scope == "full" and [x1, y1, x2, y2] != [0, 0, 1, 1]:
                raise ValueError(f"CSV row {line}: full requires bounds 0,0,1,1")
        reviewed_count += reviewed == "true"
    ready = bool(rows) and reviewed_count == len(rows) and pending_count == 0
    if require_reviewed and not ready:
        raise ValueError(
            f"Not ready: {reviewed_count}/{len(rows)} reviewed; {pending_count} pending"
        )
    return {"images": len(rows), "reviewed": reviewed_count,
            "pending": pending_count, "ready_for_evaluation": ready}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base = ROOT / "data/interim/yolo_ground_truth"
    try:
        names = {p.name for p in (base / "images").iterdir()
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        with (base / "image_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
            mapping = list(csv.DictReader(stream))
        mapped = [{key.strip(): value.strip() for key, value in row.items()}
                  for row in mapping]
        if (len(names) != 8 or len(mapped) != 8
                or {row["copied_filename"] for row in mapped} != names):
            raise ValueError("Expected exactly 8 images matching the mapping CSV")
        report = validate_scope(base / "evaluation_scope.csv", names,
                                require_reviewed=args.require_reviewed)
        for key, value in report.items():
            print(f"{key}: {value}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
