"""Run checkpointed Step 5 detection; default output is raw engineering data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangkok_cctv.engineering.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", dest="weights_path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", "--image-limit", dest="image_limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int,
                        help="Stop after N new checkpoints; resume the same target later")
    args = vars(parser.parse_args())
    config = args.pop("config")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = run_pipeline(config, **args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        logging.error("Detection stopped: %s", exc)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
