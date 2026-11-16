"""Checkpoint and failure tests use mocked inference and temporary datasets."""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bangkok_cctv.engineering import pipeline
from bangkok_cctv.engineering.detection_checkpoints import writer_lock
from bangkok_cctv.engineering.vehicle_detector import VehicleDetection


@pytest.fixture
def inputs(tmp_path: Path):
    rows = []
    for index, kind in enumerate(["zero", "vehicle", "broken", "missing", "model_error"]):
        path = tmp_path / f"data/raw/{index:03}/{kind}.png"
        path.parent.mkdir(parents=True)
        if kind == "broken":
            path.write_bytes(b"not an image")
        elif kind != "missing":
            value = {"zero": 0, "vehicle": 1, "model_error": 2}[kind]
            with Image.new("RGB", (40, 30), (value, 0, 0)) as image:
                image.save(path)
        rows.append({"image_path": path.relative_to(tmp_path).as_posix(),
                     "camera_id": f"{index:03}",
                     "timestamp": pd.Timestamp(f"2026-08-25 {6+index}:00")})
    manifest = tmp_path / "data/interim/manifest.parquet"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(rows[::-1]).to_parquet(manifest, index=False)
    weights = tmp_path / "models/yolov8n.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"mock weights")
    return tmp_path, manifest, weights, Path("data/interim/run")


class FakePredictor:
    def __init__(self):
        self.calls = 0
    def predict(self, image):
        self.calls += 1
        red = image.getpixel((0, 0))[0]
        if red == 2:
            raise RuntimeError("mock inference failure")
        return [] if red == 0 else [VehicleDetection(2, .8, 1, 1, 20, 20)]


def run(inputs, predictor, **kwargs):
    root, manifest, weights, output = inputs
    return pipeline.run_detection_pipeline(
        root, manifest, output, weights, batch_size=2, predictor=predictor, **kwargs
    )


def tables(inputs):
    root, _, _, output = inputs
    parts = sorted((root / output / "batches").glob("batch-*"))
    return (
        pd.concat([pd.read_parquet(p / "image_counts.parquet") for p in parts],
                  ignore_index=True),
        pd.concat([pd.read_parquet(p / "detections.parquet") for p in parts],
                  ignore_index=True),
    )


def test_zero_errors_schema_and_deterministic_order(inputs):
    report = run(inputs, FakePredictor())
    images, detections = tables(inputs)
    assert report["processed_images"] == 5 and report["detection_count"] == 1
    assert images.camera_id.tolist() == ["000", "001", "002", "003", "004"]
    assert images.status.tolist() == [
        "success", "success", "unreadable", "missing_file", "inference_error",
    ]
    assert images.iloc[0].total_vehicle_count == 0
    assert images.loc[images.status.ne("success"), pipeline.COUNTS].isna().all().all()
    assert images.timestamp.dt.tz is None and str(images.camera_id.dtype) in {"str", "object", "string"}
    assert report["successful_zero_detection_images"] == 1
    assert len(detections) == 1
    assert set(detections.image_id) <= set(images.image_id)


@pytest.mark.parametrize("failure", ["before_publish", "after_publish"])
def test_recover_partial_or_published_checkpoint_without_duplicates(inputs, monkeypatch, failure):
    real_commit = pipeline.commit_batch
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "before_publish" and calls == 2:
            stage = args[0] / ".pending-simulated-crash"
            stage.mkdir()
            (stage / "image_counts.parquet").write_bytes(b"incomplete")
            raise KeyboardInterrupt("simulated crash before commit")
        marker = real_commit(*args, **kwargs)
        if failure == "after_publish" and calls == 1:
            raise KeyboardInterrupt("simulated crash after rename")
        return marker
    monkeypatch.setattr(pipeline, "commit_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(inputs, FakePredictor())
    monkeypatch.setattr(pipeline, "commit_batch", real_commit)
    resumed = FakePredictor()
    report = run(inputs, resumed, resume=True)
    images, detections = tables(inputs)
    assert report["processed_images"] == 5
    assert resumed.calls == 1  # Only remaining valid-but-inference-error image.
    assert not images.duplicated(["run_id", "image_id"]).any()
    assert not detections.duplicated(["run_id", "image_id", "detection_index"]).any()
    assert len(images) == 5 and len(detections) == 1


def test_complete_resume_performs_no_inference(inputs):
    run(inputs, FakePredictor())
    unused = FakePredictor()
    result = run(inputs, unused, resume=True)
    assert unused.calls == 0 and result["processed_images"] == 5


@pytest.mark.parametrize("change", ["manifest", "weights", "limit", "batch", "configuration", "code"])
def test_incompatible_resume_refused(inputs, change):
    run(inputs, FakePredictor(), max_batches=1)
    root, manifest, weights, output = inputs
    kwargs = {}
    if change == "manifest":
        frame = pd.read_parquet(manifest)
        frame.loc[0, "camera_id"] = "changed"
        frame.to_parquet(manifest, index=False)
    elif change == "weights":
        weights.write_bytes(b"different weights")
    elif change == "limit":
        kwargs["image_limit"] = 3
    elif change in {"configuration", "code"}:
        metadata = root / output / "run.json"
        data = json.loads(metadata.read_text())
        if change == "configuration":
            data["signature"]["baseline"]["confidence"] = .05
        else:
            data["signature"]["implementation_sha256"]["pipeline.py"] = "old code"
        metadata.write_text(json.dumps(data))
    if change == "batch":
        with pytest.raises(ValueError, match="Incompatible"):
            pipeline.run_detection_pipeline(root, manifest, output, weights,
                batch_size=3, predictor=FakePredictor(), resume=True)
    else:
        with pytest.raises(ValueError, match="Incompatible"):
            run(inputs, FakePredictor(), resume=True, **kwargs)


def test_corrupt_checkpoint_refused(inputs):
    run(inputs, FakePredictor(), max_batches=1)
    root, _, _, output = inputs
    first = next((root / output / "batches").glob("batch-*"))
    with (first / "detections.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run(inputs, FakePredictor(), resume=True)


def test_clean_budgeted_resume_and_stale_summary_recovery(inputs):
    partial = run(inputs, FakePredictor(), max_batches=1)
    assert partial["status"] == "in_progress" and partial["processed_images"] == 2
    root, _, _, output = inputs
    (root / output / "summary.json").write_text("{}")
    final = run(inputs, FakePredictor(), resume=True)
    assert final["status"] == "complete" and final["processed_images"] == 5


def test_duplicate_manifest_and_existing_output_refused(inputs):
    run(inputs, FakePredictor(), max_batches=1)
    with pytest.raises(FileExistsError):
        run(inputs, FakePredictor())
    root, manifest, _, _ = inputs
    frame = pd.read_parquet(manifest)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="duplicate image"):
        run(inputs, FakePredictor(), resume=True)


def test_single_writer_and_release_after_exception(tmp_path):
    with writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with writer_lock(tmp_path):
                pass
    with writer_lock(tmp_path):
        pass


def test_all_zero_detection_parts_keep_explicit_schema(inputs):
    root, manifest, _, _ = inputs
    frame = pd.read_parquet(manifest).sort_values("camera_id").iloc[:1]
    frame.to_parquet(manifest, index=False)
    report = run(inputs, FakePredictor())
    images, detections = tables(inputs)
    assert report["successful_zero_detection_images"] == 1
    assert detections.empty and set(detections.columns) == set(pipeline.DETECTION_SCHEMA.names)
    assert images.total_vehicle_count.tolist() == [0]


def test_legacy_config_api_uses_new_pipeline(inputs):
    root, manifest, weights, output = inputs
    config = root / "config.yaml"
    config.write_text(
        f"paths:\n  model_weights: {weights.as_posix()}\n"
        f"detection:\n  manifest: {manifest.as_posix()}\n"
        f"  output_dir: {output.as_posix()}\n  batch_size: 2\n"
    )
    result = pipeline.run_pipeline(config, project_root=root, predictor=FakePredictor())
    assert result["processed_images"] == 5


def test_model_load_failure_is_fatal_and_can_resume_without_fake_counts(inputs, monkeypatch):
    root, manifest, weights, output = inputs
    def unavailable(*args, **kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(pipeline, "YOLOVehicleDetector", unavailable)
    with pytest.raises(RuntimeError, match="model unavailable"):
        pipeline.run_detection_pipeline(root, manifest, output, weights, batch_size=2)
    assert list((root / output / "batches").glob("batch-*")) == []
    fake = FakePredictor()
    monkeypatch.setattr(pipeline, "YOLOVehicleDetector", lambda *a, **k: fake)
    result = pipeline.run_detection_pipeline(
        root, manifest, output, weights, batch_size=2, resume=True
    )
    assert result["processed_images"] == 5


def test_unsafe_image_path_is_an_explicit_failed_record(inputs):
    root, manifest, _, _ = inputs
    frame = pd.read_parquet(manifest).sort_values("camera_id").iloc[:1].copy()
    frame["image_path"] = "../outside.png"
    frame.to_parquet(manifest, index=False)
    result = run(inputs, FakePredictor())
    images, detections = tables(inputs)
    assert images.iloc[0].status == "invalid_path"
    assert images.total_vehicle_count.isna().all() and detections.empty
    assert result["processed_images"] == 1
