#!/usr/bin/env python3
"""Print an aggregate summary of all recorded runs.

Usage:
    python scripts/analyze.py [--config config/study.yaml]
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from grant_study.analysis import format_summary, summarize
from grant_study.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(format_summary(summarize(config.data_dir), config.currency))


if __name__ == "__main__":
    main()
