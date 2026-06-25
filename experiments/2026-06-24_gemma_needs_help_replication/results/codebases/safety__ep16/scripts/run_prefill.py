#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only, in scope).

Prereq: a vanilla Gemma-27B-it eval run (for high-frustration seed conversations):
  results/responses/gemma-3-27b-it.rollouts.jsonl

Usage:
  python scripts/run_prefill.py
  python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESPONSES_DIR
from src.prefill.prefill_eval import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-rollouts", default=str(RESPONSES_DIR / "gemma-3-27b-it.rollouts.jsonl"))
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()

    run_prefill_experiment(Path(args.seed_rollouts), models=args.models, load_in_4bit=args.four_bit)


if __name__ == "__main__":
    main()
