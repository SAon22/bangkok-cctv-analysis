"""Entry point for the planned analysis workflow."""

from pathlib import Path

from bangkok_cctv.analysis.eda import run_eda


def main() -> None:
    """Invoke the placeholder workflow from the repository root."""
    run_eda(Path("data/processed"))


if __name__ == "__main__":
    main()
