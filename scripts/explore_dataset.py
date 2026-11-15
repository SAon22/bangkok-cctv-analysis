"""Inspect CCTV metadata and already extracted images without modifying them."""

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/raw/cctv_info.csv"
IMAGE_DIR = ROOT / "data/raw/cctv_images"
REPORT_PATH = ROOT / "data/interim/dataset_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LOG = logging.getLogger(__name__)


def explore(report):
    """Populate the report, preserving partial results if a later step fails."""
    frame = pd.read_csv(CSV_PATH, dtype={"camera_id": "string"})
    missing_columns = {"camera_id", "image_count"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {sorted(missing_columns)}")

    # Preserve leading zeros and remove accidental surrounding whitespace.
    frame["camera_id"] = frame["camera_id"].str.strip().replace("", pd.NA)
    ids = frame["camera_id"]
    counts = pd.to_numeric(frame["image_count"], errors="coerce")
    valid_counts = counts.notna() & counts.ge(0) & counts.mod(1).eq(0)
    duplicate_ids = sorted(ids[ids.notna() & ids.duplicated(keep=False)].unique())
    csv_ids = set(ids.dropna())
    statistics = json.loads(counts.describe().to_json())
    report["csv"] = {
        "shape": list(frame.shape),
        "columns": list(frame.columns),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "duplicate_camera_ids": duplicate_ids,
        "invalid_camera_id_rows": [
            int(index) for index in frame.index[ids.isna()]
        ],
        "invalid_image_count_rows": [
            int(index) for index in frame.index[~valid_counts]
        ],
        "image_count_statistics": statistics,
    }
    LOG.info("CSV shape: %s", frame.shape)
    LOG.info("Columns: %s", list(frame.columns))
    LOG.info("Missing values:\n%s", frame.isna().sum().to_string())
    LOG.info("Duplicate camera IDs: %s", duplicate_ids)
    LOG.info("image_count statistics:\n%s", counts.describe().to_string())
    if ids.isna().any() or (~valid_counts).any():
        LOG.warning("Invalid metadata rows: %s", {
            "camera_id": report["csv"]["invalid_camera_id_rows"],
            "image_count": report["csv"]["invalid_image_count_rows"],
        })

    image_dir = IMAGE_DIR
    # Some extraction tools preserve another cctv_images folder inside the target.
    if (image_dir / "cctv_images").is_dir():
        image_dir = image_dir / "cctv_images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    report["sources"]["images"] = image_dir.relative_to(ROOT).as_posix()

    actual_counts = Counter()
    folder_ids = set()
    sample_paths = []
    ignored_files = 0
    for directory in sorted(image_dir.iterdir()):
        if not directory.is_dir() or not (
            directory.name.isascii() and directory.name.isdigit()
        ):
            if directory.is_file():
                ignored_files += 1
            continue
        camera_id = directory.name
        folder_ids.add(camera_id)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored_files += 1
                continue
            actual_counts[camera_id] += 1
            if len(sample_paths) < 10:
                sample_paths.append(path)

    report["images"] = {
        "camera_count": len(folder_ids),
        "image_count": sum(actual_counts.values()),
        "image_counts_by_camera": {
            camera_id: actual_counts[camera_id] for camera_id in sorted(folder_ids)
        },
        "ignored_file_count": ignored_files,
        "sample_filenames": [
            path.relative_to(ROOT).as_posix() for path in sample_paths
        ],
    }
    LOG.info("Image cameras: %d; images: %d; ignored files: %d",
             len(folder_ids), sum(actual_counts.values()), ignored_files)
    LOG.info("Images per camera:\n%s", json.dumps(
        report["images"]["image_counts_by_camera"], indent=2
    ))
    LOG.info("Sample image paths (up to 10):")
    for path in sample_paths:
        LOG.info("  %s", path)

    mismatches = []
    # Compare each valid row; duplicate IDs are flagged, never silently aggregated.
    for index in frame.index[ids.notna() & valid_counts]:
        camera_id = str(ids.loc[index])
        expected = int(counts.loc[index])
        actual = actual_counts[camera_id]
        if expected != actual:
            mismatches.append({
                "csv_row_index": int(index), "camera_id": camera_id,
                "expected": expected, "actual": actual,
            })
    report["comparison"] = {
        "cameras_missing_from_images": sorted(csv_ids - folder_ids),
        "cameras_missing_from_csv": sorted(folder_ids - csv_ids),
        "image_count_mismatches": mismatches,
    }
    for name, findings in report["comparison"].items():
        if findings:
            LOG.warning("%s: %s", name, findings)
        else:
            LOG.info("%s: none", name)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Install it with: python -m pip install Pillow"
        ) from exc

    samples = []
    for path in sample_paths[:5]:
        sample = {"path": path.relative_to(ROOT).as_posix()}
        try:
            with Image.open(path) as image:
                sample.update(width=image.width, height=image.height,
                              format=image.format)
                image.load()  # Decode pixels to detect truncated sampled images.
            LOG.info("Sample image: %s", sample)
        except Exception as exc:
            # Continue inspecting the other samples if one image is broken.
            sample["error"] = f"{type(exc).__name__}: {exc}"
            LOG.warning("Unreadable sample %s: %s", path, exc)
        samples.append(sample)
    report["sample_images"] = samples
    report["unreadable_sample_images"] = [
        sample for sample in samples if "error" in sample
    ]
    LOG.info("Image readability checks cover only the %d sampled images.",
             len(samples))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = {
        "status": "incomplete",
        "sources": {
            "csv": CSV_PATH.relative_to(ROOT).as_posix(),
            "images": IMAGE_DIR.relative_to(ROOT).as_posix(),
        },
        "errors": [],
    }
    exit_code = 0
    try:
        explore(report)
        report["status"] = "complete"
    except (OSError, ValueError, RuntimeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        LOG.error("Exploration could not finish: %s", exc)
        exit_code = 1

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        LOG.info("Report saved to %s", REPORT_PATH)
    except OSError as exc:
        LOG.error("Could not save report to %s: %s", REPORT_PATH, exc)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
