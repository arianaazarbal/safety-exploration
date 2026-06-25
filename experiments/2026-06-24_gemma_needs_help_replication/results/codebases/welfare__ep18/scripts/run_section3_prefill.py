"""Run the Section 3 base-vs-instruct prefill experiment (Gemma only).

python -m scripts.run_section3_prefill --continuations 50
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = run_prefill_experiment(
        model_keys=tuple(args.models), seed=args.seed, continuations=args.continuations,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
