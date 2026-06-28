#!/usr/bin/env python3
"""Aggregate and print results for a completed run.

    python scripts/analyze_results.py --run data/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grant_study.analysis import format_summary, summarize_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a study run.")
    parser.add_argument("--run", required=True, help="Path to a run directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    summary = summarize_run(args.run)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
