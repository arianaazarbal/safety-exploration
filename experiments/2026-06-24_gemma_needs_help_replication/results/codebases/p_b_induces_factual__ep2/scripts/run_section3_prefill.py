#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma, scoped).

Usage:
  python scripts/run_section3_prefill.py
"""
from __future__ import annotations

import argparse
import logging

from emostab.config import load_config
from emostab.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    summary = run_prefill_experiment(cfg)
    for model, groups in summary.items():
        print(f"\n== {model} ==")
        for key, stats in groups.items():
            print(f"  {key:16s} mean={stats['mean']:.2f}  %>=5={stats['pct_high']*100:5.1f}%  (n={stats['n']})")


if __name__ == "__main__":
    main()
