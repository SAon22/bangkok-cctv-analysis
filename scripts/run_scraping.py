"""Entry point for the planned scraping workflow."""

from pathlib import Path

from bangkok_cctv.scraping.pipeline import run_pipeline


def main() -> None:
    """Invoke the placeholder workflow from the repository root."""
    run_pipeline(Path("configs/config.yaml"))


if __name__ == "__main__":
    main()
