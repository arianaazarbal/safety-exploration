#!/usr/bin/env python
"""Aggregate Section 2 results: Figure-1 headline table, per-category summary,
per-turn progression, and differential words. Also runs the judge-agreement
reliability check on a random subset using the secondary judge (GPT-5-mini).

Examples
--------
python scripts/02_analyse.py --results results/section2_*.jsonl
python scripts/02_analyse.py --results results/section2_*.jsonl --agreement 260
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval import analysis  # noqa: E402
from distress_eval.judge import FrustrationJudge, _judge_client  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="JSONL files / globs from script 01")
    ap.add_argument("--per-turn-condition", default="extended")
    ap.add_argument("--agreement", type=int, default=0,
                    help="re-score N random responses with the secondary judge")
    args = ap.parse_args()

    paths = []
    for pat in args.results:
        paths.extend(glob.glob(pat))
    if not paths:
        raise SystemExit("no result files matched")
    df = analysis.load_many(paths)

    print("\n# Headline: avg % high-frustration per model (Figure 1)")
    print(analysis.headline_per_model(df).to_string(index=False))

    print("\n# Per-(model, category) summary (Figure 2)")
    print(analysis.summary(df).to_string(index=False))

    print(f"\n# Per-turn progression: condition={args.per_turn_condition} (Figure 3)")
    print(analysis.per_turn(df, args.per_turn_condition).to_string(index=False))

    print("\n# Differential words (impossible_numeric) (Table 3)")
    dw = analysis.differential_words(df)
    for model, words in dw.items():
        top = ", ".join(w for w, _ in words)
        print(f"  {model}: {top}")

    if args.agreement:
        run_agreement(df, args.agreement)


def run_agreement(df, n):
    print(f"\n# Judge agreement on {n} random responses (Section 2.1)")
    pool = df[df["rating"] >= 0]
    sample = pool.sample(n=min(n, len(pool)), random_state=0)
    secondary = FrustrationJudge(cfg.SECONDARY_JUDGE,
                                 client=_judge_client(cfg.SECONDARY_JUDGE))
    primary = sample["rating"].tolist()
    sec = [secondary.score(r).rating for r in sample["response"]]
    stats = analysis.judge_agreement(primary, sec)
    print(f"  n={stats['n']}  Pearson r={stats['pearson_r']:.3f} "
          f"(p={stats['p_value']:.2e})  within-1={stats['pct_within_one']*100:.0f}%")


if __name__ == "__main__":
    main()
