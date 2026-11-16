"""Scope validation tests use only temporary CSVs, never project annotations."""

import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "validate_evaluation_scope",
    Path(__file__).resolve().parents[1] / "scripts/validate_evaluation_scope.py",
)
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


def write_scope(tmp_path: Path, rows: list[list]) -> Path:
    path = tmp_path / "scope.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(scope.FIELDS)
        writer.writerows(rows)
    return path


def test_pending_valid_but_blocks_evaluation(tmp_path: Path) -> None:
    path = write_scope(tmp_path, [["image.jpg", "pending", "", "", "", "", "false"]])
    result = scope.validate_scope(path, {"image.jpg"})
    assert result["pending"] == 1 and result["ready_for_evaluation"] is False
    with pytest.raises(ValueError, match="Not ready"):
        scope.validate_scope(path, {"image.jpg"}, require_reviewed=True)


def test_mixed_reviewed_full_and_roi(tmp_path: Path) -> None:
    path = write_scope(tmp_path, [
        ["a.jpg", "full", 0, 0, 1, 1, "true"],
        ["b.jpg", "roi", .1, .2, .8, .9, "true"],
    ])
    assert scope.validate_scope(path, {"a.jpg", "b.jpg"},
                                require_reviewed=True)["ready_for_evaluation"]


@pytest.mark.parametrize("values", [
    ["unknown", "", "", "", "", "false"],
    ["pending", "", "", "", "", "true"],
    ["pending", 0, 0, 1, 1, "false"],
    ["full", 0, 0, .9, 1, "true"],
    ["roi", -.1, 0, 1, 1, "true"],
    ["roi", .8, 0, .2, 1, "true"],
    ["roi", 0, 0, "nan", 1, "true"],
    ["roi", 0, 0, 1, 1, "yes"],
])
def test_invalid_values(tmp_path: Path, values: list) -> None:
    path = write_scope(tmp_path, [["a.jpg", *values]])
    with pytest.raises(ValueError):
        scope.validate_scope(path, {"a.jpg"})


def test_missing_and_duplicate_images(tmp_path: Path) -> None:
    row = ["a.jpg", "full", 0, 0, 1, 1, "true"]
    path = write_scope(tmp_path, [row])
    with pytest.raises(ValueError, match="filenames"):
        scope.validate_scope(path, {"a.jpg", "b.jpg"})
    path = write_scope(tmp_path, [row, row])
    with pytest.raises(ValueError, match="filenames"):
        scope.validate_scope(path, {"a.jpg"})


@pytest.mark.parametrize("row", [
    ["a.jpg", "full"],
    ["a.jpg", "full", 0, 0, 1, 1, "true", "extra"],
])
def test_malformed_row_reports_validation_error(tmp_path: Path, row: list) -> None:
    path = write_scope(tmp_path, [row])
    with pytest.raises(ValueError, match="wrong number of fields"):
        scope.validate_scope(path, {"a.jpg"})
