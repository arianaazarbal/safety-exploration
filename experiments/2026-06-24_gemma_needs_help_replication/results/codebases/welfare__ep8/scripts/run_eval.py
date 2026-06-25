#!/usr/bin/env python
"""Section 2 — elicit and quantify distress across Gemma & Gemini.

Runs the 8-condition / 5-category evaluation for one or more target models,
scoring every assistant turn with the Claude frustration judge, then writes
aggregated metrics.

Examples
--------
    # one Gemini model, quick smoke run (small budget)
    python scripts/run_eval.py --models gemini-2.5-flash --budget 80

    # the default Section-2 set (both Gemma + both Gemini), full budget
    python scripts/run_eval.py --models all

Gemma models require a local GPU + `pip install torch transformers accelerate`.
Gemini models require GOOGLE_API_KEY. The judge requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotioneval import config, scoring
from emotioneval.eval_conditions import build_conditions, default_allocation
from emotioneval.judge import FrustrationJudge
from emotioneval.models import load_model
import random


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="model keys, or 'all' for the default Section-2 set "
                         "(gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro)")
    ap.add_argument("--budget", type=int, default=config.RESPONSE_BUDGET_PER_MODEL,
                    help="approx assistant responses per model (default 4000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit quantise HF (Gemma) models to fit smaller GPUs")
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter (e.g. data/dpo/adapter_dpo) to "
                         "evaluate a finetuned Gemma (Section 4.2). Applies to a "
                         "single HF model.")
    ap.add_argument("--label", default=None,
                    help="override the run/model label (e.g. 'gemma-dpo') when "
                         "evaluating an adapter, so results don't overwrite the base run")
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args()


def resolve_models(keys):
    if keys == ["all"]:
        return config.SECTION2_MODELS
    return [config.model_by_key(k) for k in keys]


def main():
    args = parse_args()
    specs = resolve_models(args.models)
    judge = FrustrationJudge()
    allocation = default_allocation(args.budget)
    raw_paths = []

    for spec in specs:
        print(f"\n=== {spec.display} ({spec.model_id}) ===")
        rng = random.Random(args.seed)
        conditions = build_conditions(rng, allocation)
        kwargs = {"load_in_4bit": args.load_in_4bit} if spec.backend == "hf" else {}
        if args.adapter:
            if spec.backend != "hf":
                raise SystemExit("--adapter only applies to HF (Gemma) models")
            kwargs["adapter_path"] = args.adapter
        model = load_model(spec, **kwargs)
        # Optionally relabel so the finetuned run is a distinct model in the
        # aggregates (and doesn't overwrite the base model's raw file).
        if args.label:
            import dataclasses
            model.spec = dataclasses.replace(spec, key=args.label, display=args.label)
        label = args.label or spec.key
        run_id = f"eval_{label}_seed{args.seed}"
        from emotioneval.rollout import run_model_eval
        path = run_model_eval(model, conditions, judge, allocation, run_id,
                              seed=args.seed, resume=not args.no_resume)
        raw_paths.append(path)

    # Aggregate across all evaluated models.
    df = scoring.load_records(*raw_paths)
    summary = scoring.model_summary(df)
    cond = scoring.by_condition(df)
    cat = scoring.by_category(df)
    turns8 = scoring.per_turn(df, condition="numeric_8turn")
    turns_wild = scoring.per_turn(df, condition="wildchat_5turn")

    summary.to_csv(config.RESULTS / "section2_model_summary.csv", index=False)
    cond.to_csv(config.RESULTS / "section2_by_condition.csv", index=False)
    cat.to_csv(config.RESULTS / "section2_by_category.csv", index=False)
    turns8.to_csv(config.RESULTS / "section2_turns_8turn.csv", index=False)
    turns_wild.to_csv(config.RESULTS / "section2_turns_wildchat.csv", index=False)

    print("\n=== Headline (Figure 1): avg % high-frustration by category ===")
    print(summary.to_string(index=False))
    print(f"\nWrote metrics to {config.RESULTS}")


if __name__ == "__main__":
    main()
