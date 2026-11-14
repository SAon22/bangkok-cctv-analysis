"""Basic import checks for the analysis scaffold."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("module_name", [
    "analysis.eda",
    "analysis.temporal",
    "analysis.spatial",
    "analysis.insights"
])
def test_module_imports(module_name: str) -> None:
    """Modules should import without datasets, weights, or network access."""
    assert import_module(f"bangkok_cctv.{module_name}") is not None
