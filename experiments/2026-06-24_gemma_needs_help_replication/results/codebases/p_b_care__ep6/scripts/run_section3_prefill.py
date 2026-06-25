#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling study (Gemma 27B base vs instruct).

    python scripts/run_section3_prefill.py
"""

import argparse
import json

import _bootstrap  # noqa: F401

import config
from emotional_instability.prefill.run_prefill import run_prefill_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="+", default=list(config.PREFILL.models))
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    model_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    report = run_prefill_study(
        seed_model=args.seed_model,
        models=tuple(args.models),
        seed=args.seed,
        model_kwargs=model_kwargs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
