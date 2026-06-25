#!/usr/bin/env python3
"""Run the Section 2 distress evaluation suite (Gemma + Gemini).

Examples:
    python scripts/run_section2.py                 # full budget, scoped models
    python scripts/run_section2.py --scale 0.02     # quick smoke run
"""

from __future__ import annotations

import argparse

from gemma_needs_help import config
from gemma_needs_help.eval.run_eval import run_section2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the per-category budget")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    summary = run_section2(scale=args.scale, seed=args.seed)
    print("\n=== Figure 1 headline (avg % high-frustration) ===")
    print(summary["headline"].to_string(index=False))
    print(f"\nResults written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
