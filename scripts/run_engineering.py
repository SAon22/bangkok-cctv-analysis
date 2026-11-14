"""Entry point for the planned engineering workflow."""

from pathlib import Path

from bangkok_cctv.engineering.pipeline import run_pipeline


def main() -> None:
    """Invoke the placeholder workflow from the repository root."""
    run_pipeline(Path("configs/config.yaml"))


if __name__ == "__main__":
    main()
