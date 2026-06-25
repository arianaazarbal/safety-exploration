#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma 27B).

Requires an existing Gemma-3-27B-it elicitation run (scripts/01) to mine
high-frustration seed conversations from.

Example
-------
python scripts/02_run_prefill_experiment.py \
    --seed-results outputs/eval/gemma-3-27b-it.jsonl \
    --models gemma-3-27b-pt gemma-3-27b-it --n-continuations 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.prefill.run_prefill import run_prefill_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-results", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--n-continuations", type=int, default=50)
    args = ap.parse_args()

    run_prefill_experiment(
        seed_results_path=args.seed_results,
        target_models=args.models,
        continuations_per_prefill=args.n_continuations,
    )


if __name__ == "__main__":
    main()
