"""Build a portable image manifest from extracted CCTV images and camera metadata."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TIMESTAMP_PATTERN = re.compile(r"_(\d{8})_(\d{6})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def parse_filename_timestamp(filename: str) -> pd.Timestamp | None:
    """Read the final date/time suffix; return None for absent or invalid dates."""
    match = TIMESTAMP_PATTERN.search(Path(filename).name)
    if match is None:
        return None
    timestamp = pd.to_datetime(
        "".join(match.groups()), format="%Y%m%d%H%M%S", errors="coerce"
    )
    return None if pd.isna(timestamp) else timestamp


def join_camera_metadata(images: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join each image to one camera without dropping images or multiplying rows."""
    required = {"camera_id", "location", "latitude", "longitude"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Camera metadata missing columns: {sorted(missing)}")
    metadata = metadata.copy()
    images = images.copy()
    for frame in (images, metadata):
        frame["camera_id"] = frame["camera_id"].astype("string").str.strip()
    if metadata["camera_id"].isna().any() or metadata["camera_id"].eq("").any():
        raise ValueError("Camera metadata contains missing camera IDs")
    duplicate_ids = metadata.loc[
        metadata["camera_id"].duplicated(keep=False), "camera_id"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate camera IDs in metadata: {list(duplicate_ids)}")
    result = images.merge(
        metadata, on="camera_id", how="left", validate="many_to_one", indicator=True
    )
    result["metadata_matched"] = result.pop("_merge").eq("both")
    unmatched = sorted(result.loc[~result["metadata_matched"], "camera_id"].unique())
    if unmatched:
        LOG.warning("Images have no camera metadata for IDs: %s", unmatched)
    return result


def summarize_manifest(manifest: pd.DataFrame) -> dict:
    """Summarize timestamps; intervals include zero-length duplicate observations."""
    valid = manifest.dropna(subset=["timestamp"]).sort_values(
        ["camera_id", "timestamp"]
    )
    intervals = valid.groupby("camera_id")["timestamp"].diff().dt.total_seconds()
    interval_frame = valid[["camera_id"]].copy()
    interval_frame["interval_seconds"] = intervals
    statistics = interval_frame.groupby("camera_id")["interval_seconds"].describe()
    # Include cameras with no valid timestamps, and use JSON null for undefined stats.
    statistics = statistics.reindex(sorted(manifest["camera_id"].unique()))
    statistics["count"] = statistics["count"].fillna(0)
    interval_stats = json.loads(statistics.to_json(orient="index"))
    duplicate_groups = (
        valid.groupby(["camera_id", "timestamp"]).size().rename("image_count")
    )
    duplicate_groups = duplicate_groups[duplicate_groups > 1]
    duplicates = [
        {"camera_id": str(camera_id), "timestamp": timestamp.isoformat(),
         "image_count": int(count)}
        for (camera_id, timestamp), count in duplicate_groups.items()
    ]
    per_date = valid["timestamp"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
    invalid = manifest.loc[manifest["timestamp"].isna(), "image_path"].tolist()
    return {
        "total_images": len(manifest),
        "camera_count": int(manifest["camera_id"].nunique()),
        "timestamp_timezone": "naive; filename clock time, no timezone conversion",
        "date_range": {
            "start": valid["timestamp"].min().isoformat() if len(valid) else None,
            "end": valid["timestamp"].max().isoformat() if len(valid) else None,
        },
        "unique_dates": len(per_date),
        "images_per_date": {date: int(count) for date, count in per_date.items()},
        "images_per_camera": {
            str(camera_id): int(count)
            for camera_id, count in
            manifest.groupby("camera_id").size().sort_index().items()
        },
        "sampling_interval_units": "seconds",
        "sampling_interval_policy": (
            "Consecutive sorted valid timestamps per camera, including zero "
            "intervals for duplicates and gaps across dates"
        ),
        "sampling_interval_statistics_per_camera": interval_stats,
        "duplicate_timestamp_group_count": len(duplicates),
        "duplicate_timestamp_image_count": int(duplicate_groups.sum()),
        "duplicate_timestamp_excess_images": int((duplicate_groups - 1).sum()),
        "duplicate_timestamps": duplicates,
        "invalid_or_missing_timestamp_count": len(invalid),
        "invalid_or_missing_timestamp_paths": invalid,
        "unmatched_metadata_image_count": int((~manifest["metadata_matched"]).sum()),
    }


def build_image_manifest(project_root: Path) -> tuple[pd.DataFrame, dict]:
    """Scan only extracted image files and join CSV metadata; source files stay intact."""
    project_root = project_root.resolve()
    image_dir = project_root / "data/raw/cctv_images"
    csv_path = project_root / "data/raw/cctv_info.csv"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    rows = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rows.append({
            "camera_id": path.parent.name,
            "timestamp": parse_filename_timestamp(path.name),
            "image_path": path.relative_to(project_root).as_posix(),
        })
    if not rows:
        raise ValueError(f"No supported image files found under {image_dir}")
    images = pd.DataFrame(rows)
    images["timestamp"] = pd.to_datetime(images["timestamp"], errors="coerce")
    metadata = pd.read_csv(csv_path, dtype={"camera_id": "string"})
    manifest = join_camera_metadata(images, metadata)
    summary = summarize_manifest(manifest)
    LOG.info("Scanned %d images from %d cameras", len(manifest),
             summary["camera_count"])
    if summary["invalid_or_missing_timestamp_count"]:
        LOG.warning("Invalid or missing timestamps: %d",
                    summary["invalid_or_missing_timestamp_count"])
    return manifest, summary


def save_image_manifest(project_root: Path) -> dict:
    """Write the manifest and an accompanying JSON summary to data/interim."""
    project_root = project_root.resolve()
    manifest, summary = build_image_manifest(project_root)
    output_dir = project_root / "data/interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "image_manifest.parquet"
    summary_path = output_dir / "image_manifest_summary.json"
    manifest.to_parquet(manifest_path, engine="pyarrow", index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOG.info("Saved manifest: %s", manifest_path)
    LOG.info("Saved summary: %s", summary_path)
    return summary
