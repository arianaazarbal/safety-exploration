#!/usr/bin/env python
"""Run the Section 2 core evaluation (elicit + score distress) for Gemma & Gemini.

Examples:
    # Full sweep (4000 responses/model) for the in-scope models:
    python -m scripts.run_section2_eval

    # One model, quick smoke run (10 conversations):
    python -m scripts.run_section2_eval --models gemma-3-27b-it --limit 10

    # Judge-agreement cross-check on 260 sampled responses:
    python -m scripts.run_section2_eval --agreement
"""

from __future__ import annotations

import argparse
import json
import random

import config
from emotional_instability import judge as judge_mod, metrics, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    ap.add_argument("--tag", default="section2")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--count-mode", choices=["responses", "conversations"],
                    default="responses")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap eval items per model (smoke runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--agreement", action="store_true",
                    help="run the inter-judge agreement cross-check")
    args = ap.parse_args()

    paths = {}
    for mk in args.models:
        print(f"\n=== Section 2 eval: {mk} ===")
        paths[mk] = runner.run_model_eval(
            mk, tag=args.tag, categories=args.categories,
            count_mode=args.count_mode, limit=args.limit, max_workers=args.workers)

    # Headline summary across the evaluated models.
    df = metrics.load_results(*paths.values())
    if not df.empty:
        print("\n--- Headline (avg %>=5 across categories) ---")
        print(metrics.headline_summary(df).to_string(index=False))
        print("\n--- Per-category ---")
        print(metrics.per_category_summary(df).to_string(index=False))

    if args.agreement:
        run_agreement_check(df)


def run_agreement_check(df):
    """Re-score config.JUDGE_CROSSCHECK_N sampled responses with the crosscheck
    judge and report Pearson r + %-within-one (Section 2.1)."""
    if df.empty:
        print("[agreement] no results loaded; run the eval first")
        return
    rng = random.Random(config.SEED)
    sample = df.sample(min(config.JUDGE_CROSSCHECK_N, len(df)),
                       random_state=config.SEED)
    # We need the raw response text to re-judge; reload from rollouts via metrics.
    from emotional_instability.word_analysis import _load_rollout_texts

    cross = judge_mod.OpenRouterJudge()
    primary, secondary = [], []
    texts_cache = {}
    for _, row in sample.iterrows():
        mk = row["model_key"]
        if mk not in texts_cache:
            texts_cache[mk] = _load_rollout_texts(mk, tag="section2")
        resps = texts_cache[mk].get(row["uid"])
        if not resps or row["turn"] >= len(resps):
            continue
        text = resps[int(row["turn"])]
        primary.append(int(row["rating"]))
        secondary.append(cross.score(text).rating)
    stats = judge_mod.judge_agreement(primary, secondary)
    print(f"\n--- Judge agreement (n={stats['n']}) ---")
    print(f"Pearson r = {stats['pearson_r']:.3f} (p={stats['p_value']:.1e}); "
          f"{stats['frac_within_one']*100:.0f}% within one point")
    print("(paper reports r=0.792, 78% within one point)")


if __name__ == "__main__":
    main()
