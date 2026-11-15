"""Sampling tests do not import a detector or download model weights."""

from pathlib import Path

import pandas as pd
import pytest

from bangkok_cctv.engineering.yolo_feasibility import (
    resolve_image_path,
    select_representative_sample,
    validate_manifest,
)


def population() -> pd.DataFrame:
    rows = []
    for camera in ["001", "002", "003"]:
        for date in ["2026-08-25", "2026-08-26"]:
            for hour in [6, 12, 18]:
                for repeat in range(2):
                    rows.append({"camera_id": camera,
                                 "timestamp": pd.Timestamp(f"{date} {hour}:00:{repeat:02}"),
                                 "image_path": f"data/raw/{camera}/{date}_{hour}_{repeat}.jpg"})
    return pd.DataFrame(rows)


def test_sampling_deterministic_and_input_order_independent() -> None:
    frame = population()
    first = select_representative_sample(frame, 9, 42)
    second = select_representative_sample(frame.sample(frac=1, random_state=7), 9, 42)
    pd.testing.assert_frame_equal(first, second)
    assert first["camera_id"].nunique() == 3
    assert first["image_path"].nunique() == 9
    assert not first.duplicated(["camera_id", "sample_date", "time_band"]).any()
    assert set(first["camera_id"]) == {"001", "002", "003"}
    assert not first.equals(select_representative_sample(frame, 9, 99))


def test_all_strata_covered_when_budget_suffices() -> None:
    sample = select_representative_sample(population(), 100)
    assert len(sample) == 18
    assert set(sample["time_band"]) == {"00-10", "11-15", "16-23"}
    assert sample["sample_date"].nunique() == 2


@pytest.mark.parametrize("case", ["missing", "duplicate", "numeric_id", "aware"])
def test_manifest_validation(case: str) -> None:
    frame = population()
    if case == "missing":
        frame = frame.drop(columns="timestamp")
    elif case == "duplicate":
        frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
    elif case == "numeric_id":
        frame["camera_id"] = 1
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
    with pytest.raises(ValueError):
        validate_manifest(frame)


def test_invalid_timestamp_remains_eligible() -> None:
    frame = population().iloc[:1].copy()
    frame["timestamp"] = pd.to_datetime([None])
    sample = select_representative_sample(frame, 1)
    assert sample.loc[0, "sample_date"] == "unknown"
    assert sample.loc[0, "time_band"] == "unknown"


def test_resolve_paths(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    assert resolve_image_path(tmp_path, "image.jpg") == image
    with pytest.raises(FileNotFoundError):
        resolve_image_path(tmp_path, "missing.jpg")
    for value in ["../escape.jpg", str(image.resolve()), "C:/other/image.jpg", ""]:
        with pytest.raises(ValueError):
            resolve_image_path(tmp_path, value)


from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from bangkok_cctv.engineering import vehicle_detector as detector_module
from bangkok_cctv.engineering.vehicle_detector import VehicleDetection
from bangkok_cctv.engineering.yolo_feasibility import (
    DETECTION_DTYPES,
    detection_frame,
    run_feasibility,
    validate_outputs,
)


def write_manifest(root: Path, cases: list[str]) -> pd.DataFrame:
    rows = []
    for number, case in enumerate(cases):
        path = root / f"data/raw/cctv_images/{number:03}/{case}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if case == "broken":
            path.write_bytes(b"not an image")
        elif case != "missing":
            value = {"zero": 0, "vehicle": 1, "error": 2, "invalid": 3}.get(case, 1)
            with Image.new("RGB", (100, 80), (value, 0, 0)) as image:
                image.save(path)
        rows.append({
            "camera_id": f"{number:03}", "timestamp": pd.Timestamp("2026-08-25 12:00"),
            "image_path": path.relative_to(root).as_posix(),
        })
    frame = pd.DataFrame(rows)
    interim = root / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(interim / "image_manifest.parquet", index=False)
    return frame


class FakePredictor:
    def predict(self, image: Image.Image) -> list[VehicleDetection]:
        value = image.getpixel((0, 0))[0]
        if value == 0:
            return []
        if value == 2:
            raise RuntimeError("mock model error")
        if value == 3:
            return [VehicleDetection(2, .8, -1, 0, 20, 20)]
        return [VehicleDetection(2, .8, 1, 2, 40, 50),
                VehicleDetection(3, .9, 50, 3, 90, 60)]


def test_sampling_only_has_no_model_dependency(tmp_path: Path, monkeypatch) -> None:
    write_manifest(tmp_path, ["vehicle", "missing"])
    constructor = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr(detector_module, "YOLOVehicleDetector", constructor)
    summary = run_feasibility(tmp_path, Path("data/interim/sample_only"),
                              sampling_only=True, sample_size=2)
    sample = pd.read_parquet(tmp_path / "data/interim/sample_only/sample_manifest.parquet")
    detections = pd.read_parquet(tmp_path / "data/interim/sample_only/detections.parquet")
    assert summary["inference_executed"] is False
    assert summary["successful_images"] == summary["zero_detection_images"] == 0
    assert summary["status_counts"] == {"skipped": 1, "missing_file": 1}
    assert detections.empty and list(detections.columns) == list(DETECTION_DTYPES)
    assert sample["total_vehicle_count"].isna().all()
    assert (tmp_path / "data/interim/sample_only/annotated_images").is_dir()
    constructor.assert_not_called()


def test_success_empty_detections_and_failures_are_preserved(tmp_path: Path) -> None:
    frame = write_manifest(tmp_path, ["zero", "vehicle", "broken", "missing", "error", "invalid"])
    source_bytes = {
        path: (tmp_path / path).read_bytes()
        for path in frame["image_path"] if (tmp_path / path).is_file()
    }
    output = tmp_path / "data/interim/run"
    summary = run_feasibility(tmp_path, output, sample_size=6, predictor=FakePredictor())
    sample = pd.read_parquet(output / "sample_manifest.parquet")
    detections = pd.read_parquet(output / "detections.parquet")
    assert len(sample) == 6 and len(detections) == 2
    assert summary["successful_images"] == 2
    assert summary["zero_detection_images"] == 1
    assert summary["failed_images"] == 4
    assert summary["status_counts"] == {
        "inference_error": 2, "success": 2, "unreadable": 1, "missing_file": 1,
    }
    assert summary["per_class_detection_counts"] == {
        "car": 1, "motorcycle": 1, "bus": 0, "truck": 0,
    }
    assert summary["annotated_images"] == 2
    assert summary["runtime"]["inference_runtime_statistics"]["count"] == 4
    assert summary["evaluation_metrics"] is None
    assert set(detections["image_id"]) <= set(sample["image_id"])
    assert sample.loc[sample["status"] != "success", "total_vehicle_count"].isna().all()
    validate_outputs(sample, detections)
    for path, contents in source_bytes.items():
        assert (tmp_path / path).read_bytes() == contents
    detections.loc[0, "image_id"] = "unknown"
    with pytest.raises(ValueError, match="outside the sample"):
        validate_outputs(sample, detections)


def test_empty_detection_schema_round_trip(tmp_path: Path) -> None:
    write_manifest(tmp_path, ["zero"])
    summary = run_feasibility(tmp_path, Path("data/interim/zero"),
                              predictor=FakePredictor(), sample_size=1)
    detections = pd.read_parquet(tmp_path / "data/interim/zero/detections.parquet")
    assert detections.empty
    assert pd.api.types.is_datetime64_any_dtype(detections["timestamp"])
    assert str(detections["confidence"].dtype) == "float64"
    assert summary["zero_detection_images"] == 1
    assert summary["status"] == "complete"


def test_model_load_failure_is_persisted(tmp_path: Path, monkeypatch) -> None:
    write_manifest(tmp_path, ["vehicle"])
    monkeypatch.setattr(detector_module, "YOLOVehicleDetector",
                        Mock(side_effect=RuntimeError("weights unavailable")))
    summary = run_feasibility(tmp_path, Path("data/interim/load_error"), sample_size=1)
    assert summary["failed_images"] == 1
    assert summary["inference_executed"] is False
    assert summary["status"] == "failed"
    assert "weights unavailable" in summary["model_error"]
    assert (tmp_path / "data/interim/load_error/summary.json").is_file()


def test_annotation_failure_preserves_predictions(tmp_path: Path, monkeypatch) -> None:
    write_manifest(tmp_path, ["vehicle"])
    monkeypatch.setattr(Image.Image, "save", Mock(side_effect=OSError("disk error")))
    summary = run_feasibility(tmp_path, Path("data/interim/annotation_error"),
                              predictor=FakePredictor(), sample_size=1)
    assert summary["successful_images"] == 1
    assert summary["detection_count"] == 2
    assert summary["annotation_errors"] == 1
    assert summary["status"] == "partial_failure"


def test_output_path_and_existing_run_protected(tmp_path: Path) -> None:
    write_manifest(tmp_path, ["zero"])
    with pytest.raises(ValueError, match="subdirectory"):
        run_feasibility(tmp_path, Path("data/raw"), sampling_only=True)
    run_feasibility(tmp_path, Path("data/interim/protected"),
                    sampling_only=True, sample_size=1)
    with pytest.raises(FileExistsError):
        run_feasibility(tmp_path, Path("data/interim/protected"),
                        sampling_only=True, sample_size=1)


@pytest.mark.parametrize("kwargs", [
    {"confidence": -1}, {"confidence": float("nan")}, {"iou": 2},
    {"image_size": 31}, {"max_detections": 0}, {"sample_size": 0},
])
def test_bad_parameters_fail_before_output(tmp_path: Path, kwargs: dict) -> None:
    write_manifest(tmp_path, ["zero"])
    with pytest.raises(ValueError):
        run_feasibility(tmp_path, Path("data/interim/bad"), sampling_only=True, **kwargs)
    assert not (tmp_path / "data/interim/bad").exists()


def test_model_adapter_filters_coco_and_keeps_original_api(tmp_path: Path, monkeypatch) -> None:
    import sys

    class Tensor:
        def __init__(self, values):
            self.values = values
        def cpu(self):
            return self
        def tolist(self):
            return self.values

    predict = Mock(return_value=[SimpleNamespace(boxes=SimpleNamespace(
        xyxy=Tensor([[1, 2, 20, 30], [0, 0, 10, 10], [3, 3, 5, 5]]),
        conf=Tensor([.8, .9, .1]), cls=Tensor([2, 0, 7]),
    ))])
    model = SimpleNamespace(names=detector_module.COCO_VEHICLES, task="detect", predict=predict)
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=constructor))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"mock, not loaded")
    image_path = tmp_path / "image.png"
    with Image.new("RGB", (100, 80)) as image:
        image.save(image_path)
    counts = detector_module.detect_vehicles(image_path, weights)
    assert counts == {"car_count": 1, "motorcycle_count": 0, "bus_count": 0,
                      "truck_count": 0, "total_vehicle_count": 1}
    kwargs = predict.call_args.kwargs
    assert kwargs["classes"] == [2, 3, 5, 7]
    assert kwargs["conf"] == .25 and kwargs["save"] is False


def test_missing_weights_never_downloaded(tmp_path: Path, monkeypatch) -> None:
    download = Mock(side_effect=AssertionError("unexpected download"))
    monkeypatch.setattr(detector_module, "download_default_weights", download)
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        detector_module.YOLOVehicleDetector(tmp_path / "missing.pt")
    download.assert_not_called()


def test_cli_sampling_only(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    write_manifest(tmp_path, ["vehicle"])
    script = Path(__file__).resolve().parents[1] / "scripts/test_yolo_feasibility.py"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--project-root", str(tmp_path),
         "--sampling-only", "--sample-size", "1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["inference_executed"] is False


def test_detection_coordinates_and_schema() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        VehicleDetection(2, float("nan"), 0, 0, 1, 1).validate(100, 100)
    with pytest.raises(ValueError, match="Unsupported"):
        VehicleDetection(0, .8, 0, 0, 1, 1).validate(100, 100)
    with pytest.raises(ValueError, match="confidence"):
        VehicleDetection(2, 2, 0, 0, 1, 1).validate(100, 100)
    assert list(detection_frame([]).columns) == list(DETECTION_DTYPES)


@pytest.mark.parametrize("fault", ["count", "confidence", "status", "foreign_key"])
def test_output_schema_rejects_inconsistent_results(tmp_path: Path, fault: str) -> None:
    write_manifest(tmp_path, ["vehicle"])
    output = tmp_path / "data/interim/schema_check"
    run_feasibility(tmp_path, output, sample_size=1, predictor=FakePredictor())
    sample = pd.read_parquet(output / "sample_manifest.parquet")
    detections = pd.read_parquet(output / "detections.parquet")
    if fault == "count":
        sample.loc[0, "total_vehicle_count"] = 99
    elif fault == "confidence":
        detections.loc[0, "confidence"] = float("nan")
    elif fault == "status":
        sample.loc[0, "status"] = "not_a_status"
    else:
        detections.loc[0, "camera_id"] = "other"
    with pytest.raises(ValueError):
        validate_outputs(sample, detections)
