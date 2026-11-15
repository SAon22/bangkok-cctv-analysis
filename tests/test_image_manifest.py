"""Checks for filename timestamps, safe joins, and manifest summaries."""

from pathlib import Path

import pandas as pd
import pytest

from bangkok_cctv.engineering.image_manifest import (
    build_image_manifest,
    join_camera_metadata,
    parse_filename_timestamp,
    summarize_manifest,
)


@pytest.mark.parametrize("extension", ["jpg", "JPEG", "png"])
def test_parse_timestamp(extension: str) -> None:
    name = f"1028_???????????_???????????_20260825_061012.{extension}"
    assert parse_filename_timestamp(name) == pd.Timestamp("2026-08-25 06:10:12")


@pytest.mark.parametrize("name", [
    "1_location.jpg", "1_20260230_061012.jpg", "1_20260825_251012.jpg",
    "1_20260825_061012_extra.jpg", "1_20260825_061012.txt",
])
def test_invalid_timestamp(name: str) -> None:
    assert parse_filename_timestamp(name) is None


def metadata() -> pd.DataFrame:
    return pd.DataFrame({
        "camera_id": ["001"], "location": ["Junction"],
        "latitude": [13.7], "longitude": [100.5],
    })


def test_join_preserves_images_and_string_ids() -> None:
    images = pd.DataFrame({"camera_id": ["001", "001", "999"]})
    joined = join_camera_metadata(images, metadata())
    assert len(joined) == 3
    assert joined["camera_id"].tolist() == ["001", "001", "999"]
    assert joined["metadata_matched"].tolist() == [True, True, False]
    assert joined.loc[0, "location"] == "Junction"
    assert pd.isna(joined.loc[2, "latitude"])


def test_join_rejects_duplicate_metadata() -> None:
    with pytest.raises(ValueError, match="Duplicate camera IDs"):
        join_camera_metadata(
            pd.DataFrame({"camera_id": ["001"]}),
            pd.concat([metadata(), metadata()], ignore_index=True),
        )


def test_join_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        join_camera_metadata(pd.DataFrame({"camera_id": ["001"]}),
                             pd.DataFrame({"camera_id": ["001"]}))


def test_scan_and_summary(tmp_path: Path) -> None:
    camera_dir = tmp_path / "data/raw/cctv_images/cctv_images/001"
    camera_dir.mkdir(parents=True)
    names = ["001_a_20260825_061012.jpg", "001_b_20260825_061012.PNG",
             "001_20260825_062012.jpeg", "invalid.jpg"]
    for name in names:
        (camera_dir / name).write_bytes(b"metadata scan does not decode images")
    (camera_dir / "ignore.txt").write_text("ignore")
    metadata().to_csv(tmp_path / "data/raw/cctv_info.csv", index=False)
    manifest, summary = build_image_manifest(tmp_path)
    assert len(manifest) == 4
    assert manifest["camera_id"].tolist() == ["001"] * 4
    assert all(not Path(path).is_absolute() for path in manifest["image_path"])
    assert summary["images_per_date"] == {"2026-08-25": 3}
    assert summary["invalid_or_missing_timestamp_count"] == 1
    assert summary["duplicate_timestamp_group_count"] == 1
    assert summary["duplicate_timestamp_image_count"] == 2
    assert summary["duplicate_timestamp_excess_images"] == 1
    stats = summary["sampling_interval_statistics_per_camera"]["001"]
    assert stats["count"] == 2
    assert stats["mean"] == 300


def test_all_invalid_timestamps() -> None:
    frame = pd.DataFrame({
        "camera_id": ["001"], "timestamp": pd.to_datetime([None]),
        "image_path": ["invalid.jpg"], "metadata_matched": [True],
    })
    summary = summarize_manifest(frame)
    assert summary["date_range"] == {"start": None, "end": None}
    assert summary["sampling_interval_statistics_per_camera"]["001"]["count"] == 0
