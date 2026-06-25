#!/usr/bin/env python
"""Aggregate Section 2 outputs into the paper's headline tables (Figures 1-3, Table 3)
and optionally run the judge-agreement cross-check (Section 2.1).

Examples:
  python scripts/run_analysis.py --inputs artifacts/section2/*.jsonl
  python scripts/run_analysis.py --inputs artifacts/section2/*.jsonl --differential gemma-3-27b-it
  python scripts/run_analysis.py --inputs artifacts/section2/*.jsonl --judge-agreement
"""
import argparse
import glob

import _bootstrap  # noqa: F401

from emotional_instability.analysis import (
    differential_words,
    judge_agreement,
    load_scores,
    per_turn_progression,
    summarise_by_category,
    summarise_by_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Section 2 JSONL files (globs ok)")
    ap.add_argument("--differential", default=None, help="model name for Table 3 word analysis")
    ap.add_argument("--judge-agreement", action="store_true")
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    df = load_scores(paths)
    if df.empty:
        print("No scored rows found.")
        return

    print("\n=== Figure 1: avg % high-frustration + mean per model ===")
    print(summarise_by_model(df).to_string(index=False))

    print("\n=== Figure 2: mean + %>=5 per model x category ===")
    print(summarise_by_category(df).to_string(index=False))

    print("\n=== Figure 3: per-turn progression (extended + wildchat) ===")
    prog = per_turn_progression(df, conditions=["extended_8turn", "wildchat_5turn"])
    print(prog.to_string(index=False))

    if args.differential:
        print(f"\n=== Table 3: differential words for {args.differential} ===")
        print(differential_words(df, args.differential).to_string(index=False))

    if args.judge_agreement:
        from emotional_instability.config import load_all
        from emotional_instability.eval.judge import FrustrationJudge
        from emotional_instability.models import build_client

        registry, _ = load_all()
        second = FrustrationJudge(build_client(registry.graders["judge_agreement_check"]))
        res = judge_agreement(df, second)
        print("\n=== Judge agreement (Section 2.1 cross-check) ===")
        print(f"n={res.n}  pearson_r={res.pearson_r}  p={res.p_value:.2e}  "
              f"within_one={res.pct_within_one}%")


if __name__ == "__main__":
    main()
