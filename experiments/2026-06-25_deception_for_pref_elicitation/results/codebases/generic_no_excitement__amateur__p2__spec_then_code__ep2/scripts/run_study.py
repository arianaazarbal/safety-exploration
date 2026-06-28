#!/usr/bin/env python3
"""Run the study across all configured models, then print an aggregate summary.

Finalized allocations are recorded as PENDING for human approval — this script does
NOT move money. Use scripts/approve.py to review and (in real mode, once rails are
wired) execute approved disbursements.

Usage:
    python scripts/run_study.py [--config config/study.yaml]
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from grant_study.analysis import format_summary, summarize
from grant_study.config import load_config
from grant_study.runner import run_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_study(args.config)

    print("\n" + format_summary(summarize(config.data_dir), config.currency))
    print(
        "\nFinalized allocations are PENDING human approval. "
        "Review them with: python scripts/approve.py --list"
    )


if __name__ == "__main__":
    main()
