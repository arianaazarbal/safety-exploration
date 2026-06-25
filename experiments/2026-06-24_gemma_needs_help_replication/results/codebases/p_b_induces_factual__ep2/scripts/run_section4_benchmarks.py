#!/usr/bin/env python
"""Section 4: capability-preservation benchmarks (Figure 7).

Usage:
  python scripts/run_section4_benchmarks.py
  python scripts/run_section4_benchmarks.py --dpo-adapter runs/training/dpo/adapter
"""
from __future__ import annotations

import argparse
import logging

from emostab.benchmarks import run_benchmarks
from emostab.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--sft-adapter", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    targets = {"vanilla": None}
    if args.dpo_adapter:
        targets["dpo"] = args.dpo_adapter
    if args.sft_adapter:
        targets["sft"] = args.sft_adapter

    summary = run_benchmarks(cfg, targets=targets)
    for label, suites in summary.items():
        print(f"\n== {label} ==")
        for suite, stats in suites.items():
            flag = " (sampled)" if stats.get("sampled") else ""
            print(f"  {suite:12s} acc={stats['accuracy']:.3f}  (n={stats['n']}){flag}")


if __name__ == "__main__":
    main()
