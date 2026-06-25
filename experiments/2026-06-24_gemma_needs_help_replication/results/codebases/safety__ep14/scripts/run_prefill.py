#!/usr/bin/env python
"""Section 3 prefill experiment (base vs instruct) and the Section 4.2 recovery
test.

Requires an existing Gemma-27B-it eval run (scripts/run_eval.py) to mine
high-frustration seeds from.

Examples:
  python scripts/run_prefill.py --seed-run runs/eval/gemma-3-27b-it/responses.jsonl \
      --models gemma-3-27b-pt gemma-3-27b-it

  python scripts/run_prefill.py --recovery --seed-run runs/eval/gemma-3-27b-it/responses.jsonl \
      --models gemma-3-27b-pt gemma-3-27b-it gemma-3-27b-it-dpo
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.config import load_experiments, load_models
from emotional_instability.prefill.runner import run_prefill_experiment, run_recovery_experiment
from emotional_instability.prefill.seeds import select_recovery_seeds, select_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-run", required=True, help="path to a Gemma-it responses.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--recovery", action="store_true", help="run the recovery test instead")
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    registry = load_models()
    experiments = load_experiments()
    sampling = experiments["sampling"]
    pf = experiments["prefill"]

    if args.recovery:
        seeds = select_recovery_seeds(
            args.seed_run, min_score=pf["recovery"]["min_score"], seed=args.seed)
        out = run_recovery_experiment(
            seeds, args.models, registry,
            continuations_per_prefill=args.continuations,
            tokens_before_end=pf["recovery"]["truncate_tokens_before_end"],
            sampling=sampling)
        print(f"[recovery] wrote {out}")
    else:
        seeds = select_seeds(
            args.seed_run, n_numeric=pf["numeric_seeds"], n_text=pf["text_seeds"],
            min_score=5, seed=args.seed)
        out = run_prefill_experiment(
            seeds, args.models, registry,
            continuations_per_prefill=args.continuations,
            conditions_by_domain={"numeric": pf["truncations"]["numeric"],
                                  "text": pf["truncations"]["text"]},
            sampling=sampling)
        print(f"[prefill] wrote {out}")


if __name__ == "__main__":
    main()
