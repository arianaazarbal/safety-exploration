#!/usr/bin/env python
"""Unified entrypoint for the replication.

Examples (set EMOEVAL_PRESET=smoke|medium|full first):
  python run.py elicit --models gemma-3-27b-it gemini-2.5-flash
  python run.py prefill
  python run.py finetune --step all          # calm-data -> pairs -> DPO+SFT
  python run.py petri --models gemma-3-27b-it gemma-3-27b-it-dpo
  python run.py capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo
  python run.py analyze --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash

Stages write to data/. `analyze` reads whatever has been produced so far.
Requires (depending on stage): a GPU host for local Gemma; OPENROUTER_API_KEY
for Gemini; ANTHROPIC_API_KEY for the judges.
"""
from __future__ import annotations

import argparse

import config

# Default in-scope target sets.
ELICIT_DEFAULT = ["gemma-3-27b-it", "gemma-3-12b-it",
                  "gemini-2.5-flash", "gemini-2.5-pro"]
ANALYZE_DEFAULT = ELICIT_DEFAULT + ["gemma-3-27b-it-dpo",
                                    "gemma-3-27b-it-sft-diverse"]


def cmd_elicit(args):
    from src import elicitation
    for m in args.models:
        elicitation.run_model(m, judge_responses=not args.no_judge)


def cmd_prefill(args):
    from src import prefill
    prefill.run(source_model=args.source, models=args.models or None)


def cmd_finetune(args):
    preset = config.get_preset()
    if args.step in ("calm", "all"):
        from finetune import generate_calm_data
        generate_calm_data.generate("diverse", preset)
        if args.with_teacher:
            generate_calm_data.generate("teacher", preset)
    if args.step in ("pairs", "all"):
        from finetune import build_pairs
        build_pairs.build_sft("diverse")
        build_pairs.build_dpo()
    if args.step in ("dpo", "all"):
        from finetune import train_dpo
        train_dpo.train(str(config.FINETUNE_DIR / "dpo_pairs.jsonl"),
                        str(config.FINETUNE_DIR / "dpo_adapter"))
    if args.step in ("sft", "all"):
        from finetune import train_sft
        train_sft.train(str(config.FINETUNE_DIR / "sft_diverse.jsonl"),
                        str(config.FINETUNE_DIR / "sft_diverse_adapter"))


def cmd_petri(args):
    from src import petri_eval
    for m in args.models:
        petri_eval.run(m)


def cmd_capabilities(args):
    from src import capabilities
    for m in args.models:
        capabilities.run(m)


def cmd_analyze(args):
    from src import analysis
    models = args.models
    df = analysis.load_elicitation(models)
    if not df.empty:
        print(analysis.model_summary(df).to_string(index=False))
        analysis.condition_summary(df)
        analysis.per_turn(df)
        for m in models:
            words = analysis.differential_words(df, m)
            if words:
                print(f"\n{m} differential words:\n  {', '.join(words)}")
        if args.reliability:
            analysis.judge_reliability(df)
    analysis.prefill_summary()
    analysis.petri_summary(models)
    analysis.capability_summary(models)
    analysis.make_figures(models)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("elicit", help="Section 2 elicitation sweep")
    p.add_argument("--models", nargs="+", default=ELICIT_DEFAULT)
    p.add_argument("--no-judge", action="store_true")
    p.set_defaults(func=cmd_elicit)

    p = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    p.add_argument("--source", default="gemma-3-27b-it")
    p.add_argument("--models", nargs="+", default=None)
    p.set_defaults(func=cmd_prefill)

    p = sub.add_parser("finetune", help="Section 4 calm-data/pairs/DPO/SFT")
    p.add_argument("--step", choices=["calm", "pairs", "dpo", "sft", "all"], default="all")
    p.add_argument("--with-teacher", action="store_true")
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("petri", help="Section 4 open-ended Petri elicitation")
    p.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    p.set_defaults(func=cmd_petri)

    p = sub.add_parser("capabilities", help="Section 4 capability benchmarks")
    p.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("analyze", help="Aggregate results + figures")
    p.add_argument("--models", nargs="+", default=ANALYZE_DEFAULT)
    p.add_argument("--reliability", action="store_true",
                   help="run the GPT-5-mini cross-check (extra API cost)")
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
