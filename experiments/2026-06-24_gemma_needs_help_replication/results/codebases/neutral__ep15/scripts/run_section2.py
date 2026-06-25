#!/usr/bin/env python
"""Section 2: elicit + score distress across Gemma/Gemini, then build figures.

Usage:
    python -m scripts.run_section2 --models gemma-3-27b-it gemini-2.5-flash
    python -m scripts.run_section2 --score-only      # re-score existing rollouts
    python -m scripts.run_section2 --figures-only     # rebuild figures only

Generation runs locally (Gemma) or via OpenRouter (Gemini); scoring uses the
Claude-Sonnet-4 judge. See config.py for sample budgets / PROFILE=smoke.
"""
from __future__ import annotations

import argparse

import config
from emotional_instability.eval import runner
from emotional_instability.eval import metrics as M
from emotional_instability.analysis import figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(config.TARGET_MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    if not args.figures_only:
        for model_key in args.models:
            if args.score_only:
                resp = config.RESPONSE_DIR / f"{model_key}.jsonl"
            else:
                print(f"[gen] {model_key}")
                resp = runner.generate_model(model_key, seed=args.seed)
            print(f"[score] {model_key}")
            runner.score_file(resp)

    df = M.load_all()
    if df.empty:
        print("No scored data found.")
        return
    print("\n=== Figure 1: avg % high-frustration ===")
    print(M.figure1_table(df).to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(M.figure2_table(df).to_string(index=False))

    for fn in (figures.figure1, figures.figure2, figures.figure3):
        print("wrote", fn(df))


if __name__ == "__main__":
    main()
