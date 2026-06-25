#!/usr/bin/env python3
"""CLI: compute metrics from a completed evaluation run.

Example:
  python run_analysis.py results/pilot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from distress_eval.analysis import analyse_run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analyse a distress-elicitation run.")
    p.add_argument("run_dir", type=Path, help="Run directory (e.g. results/pilot).")
    p.add_argument("--threshold", type=int, default=5, help="High-frustration score threshold.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)

    res = analyse_run(args.run_dir, threshold=args.threshold)

    print("\n=== Overall (Figure 1) ===")
    print(res["overall"].to_string(index=False))
    print("\n=== By category (Figure 2) ===")
    print(res["by_category"].to_string(index=False))
    print("\n=== Per-turn trajectory (Figure 3) ===")
    print(res["per_turn"].to_string(index=False))
    print("\n=== Coverage / data quality ===")
    print(res["coverage"].to_string(index=False))
    if res["reliability"]:
        r = res["reliability"]
        print(
            f"\n=== Judge reliability ===\nn={r['n']}, Pearson r={r['pearson_r']:.3f}, "
            f"within one point={r['pct_within_one_point']:.1f}%"
        )
    print(f"\nArtefacts written to: {res['analysis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
