"""Focused diagnostic-tool tests; all model predictions are mocked."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "diagnose_yolo", Path(__file__).resolve().parents[1] / "scripts/diagnose_yolo.py"
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    # A real Unicode filename exercises exact manifest matching and Pillow paths.
    image = tmp_path / "data/raw/cctv_images/001/\u0e41\u0e22\u0e01.png"
    image.parent.mkdir(parents=True)
    with Image.new("RGB", (52, 40), "gray") as source:
        source.save(image)
    interim = tmp_path / "data/interim"
    interim.mkdir(parents=True)
    pd.DataFrame([{
        "image_path": image.relative_to(tmp_path).as_posix(), "camera_id": "001",
        "timestamp": pd.Timestamp("2026-08-30 13:15:54"),
    }]).to_parquet(interim / "image_manifest.parquet", index=False)
    selection = tmp_path / "images.json"
    selection.write_text(json.dumps({"images": [{
        "image_path": image.relative_to(tmp_path).as_posix(),
        "note": "Suspected missed vehicle; not ground truth", "nms": False,
    }]}), encoding="utf-8")
    weights = tmp_path / "models/yolov8n.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"mock weights; never loaded")
    previous = interim / "yolo_feasibility"
    previous.mkdir()
    (previous / "summary.json").write_text(json.dumps({
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()
    }), encoding="utf-8")
    return selection, weights, image


class Tensor:
    def __init__(self, values: list):
        self.values = values
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, *, empty: bool = False, fail: bool = False):
        self.empty, self.fail = empty, fail
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail and kwargs["classes"] is None:
            raise RuntimeError("mock all-class error")
        labels, scores = [], []
        if not self.empty:
            labels, scores = [2], [.8]
            if kwargs["conf"] == .05:
                labels.append(3)
                scores.append(.1)
            if kwargs["classes"] is None:
                labels.append(0)
                scores.append(.6)
        image = kwargs["source"]
        return [SimpleNamespace(
            names={0: "person", 2: "car", 3: "motorcycle"},
            orig_shape=(image.height, image.width),
            boxes=SimpleNamespace(
                xyxy=Tensor([[1, 2, 20, 30] for _ in labels]),
                cls=Tensor(labels), conf=Tensor(scores),
            ),
        )]


def test_default_cli_only_validates(tmp_path: Path, inputs, monkeypatch, capsys) -> None:
    selection, _, _ = inputs
    constructor = Mock(side_effect=AssertionError("Inference not authorized"))
    monkeypatch.setattr(diagnostics, "YOLOVehicleDetector", constructor)
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["diagnose_yolo.py", "--images", str(selection)])
    assert diagnostics.main() == 0
    assert "No inference" in capsys.readouterr().out
    constructor.assert_not_called()
    assert not (tmp_path / "data/interim/yolo_diagnostics").exists()


@pytest.mark.parametrize("fault", ["missing_file", "absent_from_manifest", "duplicate", "too_many"])
def test_no_replacements_for_invalid_lists(tmp_path: Path, inputs, fault: str) -> None:
    selection, _, image = inputs
    data = json.loads(selection.read_text(encoding="utf-8"))
    if fault == "missing_file":
        image.unlink()
    elif fault == "absent_from_manifest":
        data["images"][0]["image_path"] = "not_selected.png"
    elif fault == "duplicate":
        data["images"] *= 2
    else:
        data["images"] *= 9
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        diagnostics.load_images(tmp_path, selection)


@pytest.mark.parametrize("optional_nms", [False, True])
def test_configurations_tables_and_original_size_panels(
    tmp_path: Path, inputs, monkeypatch, optional_nms: bool,
) -> None:
    selection, weights, image = inputs
    original = image.read_bytes()
    data = json.loads(selection.read_text(encoding="utf-8"))
    data["images"][0]["nms"] = optional_nms
    selection.write_text(json.dumps(data), encoding="utf-8")
    fake = FakeModel()
    monkeypatch.setattr(diagnostics, "YOLOVehicleDetector",
                        Mock(return_value=SimpleNamespace(model=fake)))
    output = tmp_path / "data/interim/yolo_diagnostics/comparison"
    report = diagnostics.run_diagnostics(tmp_path, selection, output, weights)
    expected_configs = diagnostics.CONFIGS if optional_nms else diagnostics.CONFIGS[:3]
    assert len(fake.calls) == len(expected_configs)
    for call, config in zip(fake.calls, expected_configs):
        assert (call["conf"], call["classes"], call["iou"]) == (
            config["confidence"], config["classes"], config["iou"]
        )
        assert call["imgsz"] == 640 and call["save"] is False
        assert call["rect"] is True and call["agnostic_nms"] is False
    table = pd.read_csv(output / "detections.csv", dtype={"camera_id": "string"})
    assert list(table.columns) == diagnostics.COLUMNS
    assert set(table.loc[table.configuration.eq("all_classes"), "class_name"]) == {
        "person", "car", "motorcycle",
    }
    assert set(table.camera_id) == {"001"}
    assert report["failed_configurations"] == 0
    assert report["image_list"][0]["note"] == data["images"][0]["note"]
    with Image.open(next(output.glob("*_comparison.png"))) as combined:
        assert combined.size == (52 * (1 + len(expected_configs)), 40 + 45)
    assert image.read_bytes() == original
    with pytest.raises(FileExistsError):
        diagnostics.run_diagnostics(tmp_path, selection, output, weights)


@pytest.mark.parametrize("empty,fail", [(True, False), (False, True)])
def test_zero_detections_and_configuration_failures(
    tmp_path: Path, inputs, monkeypatch, empty: bool, fail: bool,
) -> None:
    selection, weights, _ = inputs
    monkeypatch.setattr(diagnostics, "YOLOVehicleDetector", Mock(return_value=
        SimpleNamespace(model=FakeModel(empty=empty, fail=fail))))
    output = tmp_path / "data/interim/yolo_diagnostics/result"
    report = diagnostics.run_diagnostics(tmp_path, selection, output, weights)
    results = pd.read_csv(output / "image_results.csv")
    assert len(results) == 3
    if empty:
        assert pd.read_csv(output / "detections.csv").empty
        assert results.detection_count.eq(0).all()
        assert results.status.eq("success").all()
    else:
        failed = results.loc[results.configuration.eq("all_classes")].iloc[0]
        assert failed.status == "error" and pd.isna(failed.detection_count)
        assert "mock all-class error" in failed.error
        assert report["failed_configurations"] == 1


def test_weight_mismatch_and_output_escape_fail_before_model(
    tmp_path: Path, inputs, monkeypatch,
) -> None:
    selection, weights, _ = inputs
    constructor = Mock(side_effect=AssertionError("must not construct"))
    monkeypatch.setattr(diagnostics, "YOLOVehicleDetector", constructor)
    with pytest.raises(ValueError, match="NEW run"):
        diagnostics.run_diagnostics(tmp_path, selection, tmp_path / "data/raw", weights)
    weights.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="differ"):
        diagnostics.run_diagnostics(
            tmp_path, selection, tmp_path / "data/interim/yolo_diagnostics/bad", weights
        )
    constructor.assert_not_called()
