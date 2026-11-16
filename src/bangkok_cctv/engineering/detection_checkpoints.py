"""Atomic checkpoint directories and single-writer locking for detection runs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Flush content before atomic replacement; temporary files are never readers."""
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def writer_lock(directory: Path) -> Iterator[None]:
    """OS-released lock: a killed process does not leave an active stale lock."""
    with (directory / "run.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process is writing this run") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def commit_batch(
    directory: Path, start: int, images: pa.Table, detections: pa.Table,
    run_id: str, elapsed_seconds: float,
) -> dict:
    """Publish the two Parquet files together by renaming a same-volume directory."""
    stop = start + len(images)
    destination = directory / f"batch-{start:08d}-{stop:08d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already committed: {destination}")
    temporary = directory / f".pending-{uuid.uuid4().hex}"
    temporary.mkdir()
    for name, table in [("image_counts.parquet", images), ("detections.parquet", detections)]:
        with (temporary / name).open("xb") as stream:
            pq.write_table(table, stream, compression="snappy")
            stream.flush()
            os.fsync(stream.fileno())
    marker = {
        "start": start, "stop": stop, "run_id": run_id,
        "image_count": len(images), "detection_count": len(detections),
        "image_sha256": sha256(temporary / "image_counts.parquet"),
        "detection_sha256": sha256(temporary / "detections.parquet"),
        "processing_seconds": elapsed_seconds,
    }
    atomic_json(temporary / "commit.json", marker)
    os.replace(temporary, destination)
    return marker


def recover_batches(
    directory: Path, plan: list[dict], run_id: str,
    image_schema: pa.Schema, detection_schema: pa.Schema,
) -> tuple[int, list[dict]]:
    """Read only committed directories; reject gaps, corruption and wrong identity."""
    cursor, records = 0, []
    for batch in sorted(directory.glob("batch-*")):
        marker = json.loads((batch / "commit.json").read_text(encoding="utf-8"))
        start, stop = marker["start"], marker["stop"]
        if (marker["run_id"] != run_id or start != cursor or not start < stop <= len(plan)
                or batch.name != f"batch-{start:08d}-{stop:08d}"):
            raise ValueError(f"Noncontiguous or incompatible checkpoint: {batch.name}")
        image_path, detection_path = batch / "image_counts.parquet", batch / "detections.parquet"
        if (sha256(image_path) != marker["image_sha256"]
                or sha256(detection_path) != marker["detection_sha256"]):
            raise ValueError(f"Checkpoint checksum mismatch: {batch.name}")
        images, detections = pq.read_table(image_path), pq.read_table(detection_path)
        if (not images.schema.equals(image_schema) or not detections.schema.equals(detection_schema)
                or len(images) != stop - start or len(images) != marker["image_count"]
                or len(detections) != marker["detection_count"]):
            raise ValueError(f"Checkpoint schema/count mismatch: {batch.name}")
        rows = images.to_pylist()
        if ([row["image_path"] for row in rows]
                != [row["image_path"] for row in plan[start:stop]]
                or [row["source_index"] for row in rows] != list(range(start, stop))):
            raise ValueError(f"Checkpoint image order/identity mismatch: {batch.name}")
        parents = {row["image_id"]: row for row in rows}
        if len(parents) != len(rows):
            raise ValueError("Duplicate checkpoint image IDs")
        seen = set()
        for detection in detections.to_pylist():
            key = (detection["image_id"], detection["detection_index"])
            parent = parents.get(detection["image_id"])
            if (key in seen or parent is None or parent["status"] != "success"
                    or detection["run_id"] != run_id
                    or detection["image_path"] != parent["image_path"]):
                raise ValueError(f"Duplicate/orphan checkpoint detection: {batch.name}")
            seen.add(key)
        for actual, expected in zip(rows, plan[start:stop], strict=True):
            if (actual["run_id"] != run_id or actual["camera_id"] != expected["camera_id"]
                    or actual["timestamp"] != expected["timestamp"]):
                raise ValueError("Checkpoint metadata differs from the manifest")
        records.append(marker)
        cursor = stop
    return cursor, records
