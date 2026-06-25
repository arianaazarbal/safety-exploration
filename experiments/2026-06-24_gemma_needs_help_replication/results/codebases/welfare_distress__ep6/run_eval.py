#!/usr/bin/env python3
"""CLI for the distress-elicitation replication (Section 2 of the paper).

Examples
--------
# Quick wiring check (tiny rollout counts), Gemma + Gemini via OpenRouter:
    python run_eval.py --preset smoke

# Default moderate run over the 4 default models:
    python run_eval.py

# Full paper-scale response counts for a subset of models:
    python run_eval.py --preset paper --models gemma-3-27b-it gemini-2.5-flash

# Generate transcripts only (no judging):
    python run_eval.py --no-judge

# Analyse an existing responses.jsonl:
    python run_eval.py --analyze-only --output-dir ./outputs

Environment
-----------
  OPENROUTER_API_KEY   required for Gemma/Gemini generation
  ANTHROPIC_API_KEY    required for the Claude-Sonnet-4 judge
"""

from __future__ import annotations

import argparse
import json
import os

from distress_eval.analyze import (
    format_category_table,
    format_summary_table,
    load_records,
    per_turn,
    summarize,
)
from distress_eval.config import (
    DEFAULT_MODELS,
    GenConfig,
    JudgeSpec,
    RunConfig,
    apply_preset,
)
from distress_eval.conditions import CONDITIONS
from distress_eval.runner import run_eval


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    help="model registry keys to evaluate")
    ap.add_argument("--preset", default="default", choices=["smoke", "default", "paper"],
                    help="rollout-count preset")
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--judge-model", default=JudgeSpec().model_id)
    ap.add_argument("--no-judge", action="store_true", help="generate transcripts without scoring")
    ap.add_argument("--final-turn-only", action="store_true",
                    help="score only the final assistant turn instead of every turn")
    ap.add_argument("--no-hf-wildchat", action="store_true",
                    help="use the static WildChat fallback prompts (no datasets/network)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip generation; summarise an existing responses.jsonl")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.analyze_only:
        config = RunConfig(
            models=args.models,
            conditions=apply_preset(CONDITIONS, args.preset),
            gen=GenConfig(temperature=args.temperature, max_tokens=args.max_tokens),
            judge=JudgeSpec(model_id=args.judge_model),
            seed=args.seed,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            score_all_turns=not args.final_turn_only,
            judge_enabled=not args.no_judge,
            use_hf_wildchat=not args.no_hf_wildchat,
        )
        run_eval(config)

    records_path = os.path.join(args.output_dir, "responses.jsonl")
    if not os.path.exists(records_path):
        print(f"No records at {records_path}; nothing to analyse.")
        return

    records = load_records(records_path)
    summary = summarize(records)
    turns = per_turn(records)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "per_turn": turns}, f, indent=2)

    print("\n=== Headline: average % high-frustration (>=5) across categories (Fig. 1) ===")
    print(format_summary_table(summary))
    print("\n=== Per-category % high-frustration (Fig. 2) ===")
    print(format_category_table(summary))
    print(f"\nWrote detailed summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
