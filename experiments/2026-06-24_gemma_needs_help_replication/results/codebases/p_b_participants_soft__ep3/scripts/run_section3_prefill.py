#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma pair).

Requires Section-2 rollouts for the seed model (gemma-3-27b-it) to exist, since
seeds are high-frustration conversations drawn from them.

Example:
    python scripts/run_section3_prefill.py --load-in-4bit
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.prefill.run_prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    results = run_prefill_experiment(
        seed_model=args.seed_model,
        continuations_per_prefill=args.continuations,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
