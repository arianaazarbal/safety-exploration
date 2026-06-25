#!/usr/bin/env python3
"""Run the Section 3 base-vs-instruct prefill experiment (Gemma only).

Requires Section 2 to have been run first (it sources high-frustration seed
conversations from the Gemma-3-27B-it results).
"""

from __future__ import annotations

import argparse

from gemma_needs_help import config
from gemma_needs_help.prefill.run_prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    summary = run_prefill_experiment(seed=args.seed)
    print(summary.to_string(index=False))
    print(f"\nResults written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
