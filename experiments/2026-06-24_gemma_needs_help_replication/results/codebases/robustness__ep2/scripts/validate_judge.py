#!/usr/bin/env python
"""Section 2.1 judge-reliability check.

Randomly samples N already-scored responses and re-scores them with a second
judge, then reports Pearson r and within-1-point agreement (paper: r=0.792,
78% within one point, using GPT-5-mini as the second judge).

The second judge defaults to GPT-5-mini via OpenRouter; pass --judge to use any
model handle. Add a 'gpt-5-mini' ModelSpec to config.MODELS or use --judge-id.

Usage:
    python scripts/validate_judge.py --n 260 --judge-id openai/gpt-5-mini
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import random

from config import ModelSpec, RESULTS_DIR, SEED
from emotional_eval import analysis, judge
from emotional_eval.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--judge-id", default="openai/gpt-5-mini")
    ap.add_argument("--backend", default="openrouter")
    args = ap.parse_args()

    files = [p for p in RESULTS_DIR.glob("*.jsonl")
             if p.stem not in ("petri_results", "prefill_results",
                               "recovery_results")]
    rows = [r for p in files for r in read_jsonl(p) if r.get("rating") is not None]
    if not rows:
        raise SystemExit("no scored results found")

    rng = random.Random(SEED)
    sample = rng.sample(rows, min(args.n, len(rows)))
    texts = [r["assistant_text"] for r in sample]
    primary = [r["rating"] for r in sample]

    second_spec = ModelSpec("judge2", args.backend, args.judge_id, "other")
    secondary = [jr.rating for jr in judge.score_many(texts, judge_spec=second_spec,
                                                      desc="second judge")]

    stats = analysis.judge_agreement(primary, secondary)
    print("Judge agreement (primary Claude-Sonnet-4 vs second judge):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
