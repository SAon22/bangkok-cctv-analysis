"""Basic import checks for the engineering scaffold."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("module_name", [
    "engineering.image_loader",
    "engineering.vehicle_detector",
    "engineering.preprocess",
    "engineering.validate",
    "engineering.pipeline"
])
def test_module_imports(module_name: str) -> None:
    """Modules should import without datasets, weights, or network access."""
    assert import_module(f"bangkok_cctv.{module_name}") is not None
