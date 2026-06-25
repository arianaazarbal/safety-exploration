#!/usr/bin/env python3
"""Section 4.2: re-run the Section-2 eval on the finetunes and compare.

Evaluates gemma-3-27b-it, gemma-3-27b-dpo, gemma-3-27b-sft-diverse,
gemma-3-27b-sft-teacher and prints the Figure-5 comparison (avg % high-
frustration and per-category breakdown). Expect DPO 35% -> ~0.3%.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import FINETUNE_EVAL_MODELS, RESULTS_DIR
from src.eval.analyze import load_scores, per_category, summarize_models
from src.eval.run_eval import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=FINETUNE_EVAL_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        for m in args.models:
            print(f"\n=== Eval {m} ===")
            run_eval(m, seed=args.seed)

    print("\n=== Figure 5: avg % high-frustration (finetunes vs vanilla) ===")
    table = summarize_models(args.models)
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "figure5_finetune_headline.csv")

    for m in args.models:
        try:
            df = load_scores(m)
        except FileNotFoundError:
            continue
        print(f"\n--- {m} :: per-category ---")
        print(per_category(df).to_string())


if __name__ == "__main__":
    main()
