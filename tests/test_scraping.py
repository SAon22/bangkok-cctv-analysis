"""Basic import checks for the scraping scaffold."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("module_name", [
    "scraping.sources",
    "scraping.scraper",
    "scraping.pipeline"
])
def test_module_imports(module_name: str) -> None:
    """Modules should import without datasets, weights, or network access."""
    assert import_module(f"bangkok_cctv.{module_name}") is not None
