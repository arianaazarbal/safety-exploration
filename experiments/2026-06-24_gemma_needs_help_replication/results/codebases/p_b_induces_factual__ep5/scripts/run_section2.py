#!/usr/bin/env python
"""Section 2 — elicit + score frustration across the 8 conditions, for one model.

Usage:
    python scripts/run_section2.py --model gemma-3-27b-it
    python scripts/run_section2.py --model gemini-2.5-flash --total 4000
    python scripts/run_section2.py --model dpo --adapter checkpoints/dpo_gemma_27b

Then validate the judge on a subset and run the analysis:
    python scripts/run_section2.py --score-only results/section2/<model>.jsonl --validate 260
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from gemma_distress import config
from gemma_distress.analysis import aggregate, judge_agreement, per_turn
from gemma_distress.analysis.differential_words import differential_words
from gemma_distress.eval.runner import run_elicitation
from gemma_distress.judge import FrustrationJudge, ValidationJudge
from gemma_distress.judge.frustration_judge import score_records as _score
from gemma_distress.models.factory import load_model
from gemma_distress.storage import JsonlWriter, read_jsonl


def cmd_elicit(args):
    key = config.DPO_BASE_MODEL if args.model == "dpo" else args.model
    model = load_model(key, adapter_path=args.adapter)
    name = "dpo-gemma" if args.model == "dpo" else args.model
    model.name = name
    path = run_elicitation(model, total_responses=args.total)
    print(f"[elicit] wrote rollouts -> {path}")

    scored = _score(path, judge=FrustrationJudge())
    print(f"[score] wrote scored -> {scored}")
    return scored


def cmd_validate(scored_path: Path, n: int):
    """Re-score a random subset with GPT-5-mini and report judge agreement."""
    records = list(read_jsonl(scored_path))
    rng = random.Random(0)
    subset = rng.sample(records, min(n, len(records)))
    vj = ValidationJudge()
    for rec in subset:
        rec["validation_score"], _ = vj.score(rec.get("response", ""))
    out = Path(scored_path).with_suffix(".validated.jsonl")
    JsonlWriter(out).write_many(subset)

    import pandas as pd

    agree = judge_agreement(pd.DataFrame(subset))
    print(f"[validate] judge agreement (n={agree['n']}): {agree}")


def cmd_analyze(paths):
    df = aggregate.load_scored(paths)
    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(aggregate.figure1_table(df).to_string(index=False))
    print("\n=== Figure 2: mean frustration + % >=5 per (model, category) ===")
    print(aggregate.figure2_summary(df).to_string(index=False))
    print("\n=== Figure 3: per-turn progression (extended, wildchat) ===")
    print(per_turn.figure3_per_turn(df).to_string(index=False))
    for model in df["model"].unique():
        words = differential_words(df, model=model)
        print(f"\n=== Table 3: differential words [{model}] ===\n{', '.join(words)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model key, or 'dpo'")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (for dpo)")
    ap.add_argument("--total", type=int, default=config.SAMPLES_PER_MODEL)
    ap.add_argument("--score-only", default=None, help="score an existing rollout file")
    ap.add_argument("--validate", type=int, default=0, help="N subset for GPT-5-mini")
    ap.add_argument("--analyze", nargs="*", help="scored jsonl paths to analyse")
    args = ap.parse_args()

    if args.analyze:
        cmd_analyze(args.analyze)
        return

    if args.score_only:
        scored = _score(args.score_only, judge=FrustrationJudge())
        print(f"[score] -> {scored}")
        if args.validate:
            cmd_validate(Path(scored), args.validate)
        return

    scored = cmd_elicit(args)
    if args.validate:
        cmd_validate(Path(scored), args.validate)


if __name__ == "__main__":
    main()
