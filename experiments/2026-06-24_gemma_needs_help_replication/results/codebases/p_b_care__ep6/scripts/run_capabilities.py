#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks.

Compare a finetuned Gemma against the vanilla instruct model on AIME/MATH/GPQA/
BBH/TruthfulQA/EmoBench. Run once per model and diff the reports.

    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it-dpo --limit 50
"""

import argparse
import json

import _bootstrap  # noqa: F401

import config
from emotional_instability.capabilities.run_benchmarks import evaluate_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(config.CAPABILITY_BENCHMARKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per benchmark (for quick checks)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    model_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    report = evaluate_model(args.model, benchmarks=tuple(args.benchmarks),
                            limit=args.limit, model_kwargs=model_kwargs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
