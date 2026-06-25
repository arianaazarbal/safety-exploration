#!/usr/bin/env python
"""Section 3: base vs instruct distress via prefilled continuations (Gemma only).

Requires the Section 2 scored output for the seed model (default
gemma-3-27b-it) to exist, since seeds are drawn from its high-frustration
responses.

Usage:
    python -m scripts.run_section3
    python -m scripts.run_section3 --recovery   # Sec 4.2 recovery test instead
"""
from __future__ import annotations

import argparse

from emotional_instability.prefill.continuations import run_prefill_experiment
from emotional_instability.prefill import metrics as PM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*", default=None,
                    help="default: Gemma base + instruct")
    ap.add_argument("--recovery", action="store_true")
    args = ap.parse_args()

    out = run_prefill_experiment(seed_model=args.seed_model, models=args.models,
                                 recovery=args.recovery)
    df = PM.load(out)
    if args.recovery:
        print(PM.recovery_rate(df).to_string(index=False))
    else:
        print(PM.figure4_table(df).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
