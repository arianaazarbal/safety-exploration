"""Run the Section 2 elicitation evaluation suite.

Examples:
    python scripts/run_eval.py                       # all configured models
    python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_eval.py --profile smoke       # cheap dry run
    python scripts/run_eval.py --include-finetuned   # also eval DPO/SFT checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval import aggregate
from distress_eval.backends import get_backend
from distress_eval.config import load_config
from distress_eval.evaluation import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model keys to evaluate (default: all in config.models)")
    ap.add_argument("--profile", default=None, help="override profile (full|smoke)")
    ap.add_argument("--include-finetuned", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.profile:
        config.profile = args.profile

    pool = list(config.models)
    if args.include_finetuned:
        pool += config.finetuned_models
    if args.models:
        keys = set(args.models)
        pool = [m for m in pool if m.key in keys]

    judge_backend = get_backend(config.judge, generation=config.generation)

    for model in pool:
        print(f"\n=== Evaluating {model.key} ({model.backend}) ===")
        path = evaluate_model(config, model, judge_backend, resume=not args.no_resume)
        print(f"  -> {path}")

    # Summary
    df = aggregate.load_responses(config.output_dir / "responses")
    if not df.empty:
        summary = aggregate.per_model_summary(df, convention="final")
        print("\n=== Per-model summary (Figure 1/2) ===")
        print(summary.to_string(index=False))
        summary.to_csv(config.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
