#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks.

Run the same set on vanilla and DPO/SFT models and compare (paper: no reductions).

Example
-------
python scripts/07_run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --benchmarks aime math gpqa bbh truthfulqa emobench --limit 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.capabilities.benchmarks import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for model in args.models:
        run_all(model, benchmarks=args.benchmarks, limit=args.limit)


if __name__ == "__main__":
    main()
