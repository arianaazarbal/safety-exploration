#!/usr/bin/env python3
"""Run the grant-allocation study from a config file.

    python scripts/run_study.py --config config/study_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grant_study.config import StudyConfig  # noqa: E402
from grant_study.study import run_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grant-allocation study.")
    parser.add_argument("--config", required=True, help="Path to study YAML config.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    args = parser.parse_args()

    cfg = StudyConfig.from_yaml(args.config)
    run_study(cfg, run_id=args.run_id)


if __name__ == "__main__":
    main()
