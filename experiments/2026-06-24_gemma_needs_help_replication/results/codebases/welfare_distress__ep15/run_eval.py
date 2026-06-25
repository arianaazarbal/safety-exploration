#!/usr/bin/env python3
"""Entry point for the emotional-instability elicitation replication.

Pipeline stages (run independently or end-to-end):

  generate   sample multi-turn rollouts from each target model      -> results/rollouts/
  score      score every assistant turn with the Claude judge       -> results/scores/
  agreement  re-score a random subset with the GPT-5-mini judge      -> reliability stats
  analyze    aggregate scores into Figure 1/2/3 numbers              -> results/analysis/

Examples:
  REPLICATION_PRESET=smoke python run_eval.py all
  python run_eval.py generate --models gemini-2.5-flash
  python run_eval.py score
  python run_eval.py analyze

Scope: Gemma + Gemini target models only (config.TARGET_MODELS).
"""

from __future__ import annotations

import argparse
import json

from analysis.aggregate import build_report, print_summary, write_report
from config import TARGET_MODELS, get_sample_counts
from evals.conditions import build_rollout_specs
from harness.runner import run_model_rollouts
from models.factory import build_model
from scoring.score_runner import score_model, validate_judge_agreement


def _selected_models(names: list[str] | None):
    if not names:
        return TARGET_MODELS
    chosen = [m for m in TARGET_MODELS if m.name in names]
    missing = set(names) - {m.name for m in chosen}
    if missing:
        raise SystemExit(f"Unknown model(s): {sorted(missing)}")
    return chosen


def cmd_generate(args):
    counts = get_sample_counts()
    specs = build_rollout_specs(counts)
    print(f"Built {len(specs)} rollout specs/model "
          f"(preset={'smoke' if counts.extended < 10 else 'full'}).")
    for cfg in _selected_models(args.models):
        print(f"\n>>> Generating rollouts for {cfg.name} ({cfg.backend}:{cfg.model_id})")
        model = build_model(cfg)
        path = run_model_rollouts(model, specs, batch_size=args.batch_size)
        print(f"    wrote {path}")


def cmd_score(args):
    from models.judge import AnthropicJudge

    judge = AnthropicJudge()
    for cfg in _selected_models(args.models):
        print(f"\n>>> Scoring {cfg.name}")
        path = score_model(cfg.name, judge)
        print(f"    wrote {path}")


def cmd_agreement(args):
    from models.judge import OpenAIJudge

    names = [m.name for m in _selected_models(args.models)]
    stats = validate_judge_agreement(names, OpenAIJudge())
    print("\n=== Judge reliability (primary vs secondary) ===")
    print(json.dumps(stats, indent=2))


def cmd_analyze(args):
    names = [m.name for m in _selected_models(args.models)]
    report = build_report(names)
    json_path, csv_path = write_report(report)
    print_summary(report)
    print(f"\nWrote {json_path}\nWrote {csv_path}")


def cmd_all(args):
    cmd_generate(args)
    cmd_score(args)
    cmd_analyze(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--models", nargs="*", help="subset of target model names")

    g = sub.add_parser("generate", help="sample rollouts from target models")
    add_common(g)
    g.add_argument("--batch-size", type=int, default=8)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="score rollouts with the Claude judge")
    add_common(s)
    s.set_defaults(func=cmd_score)

    a = sub.add_parser("agreement", help="judge reliability via the secondary judge")
    add_common(a)
    a.set_defaults(func=cmd_agreement)

    an = sub.add_parser("analyze", help="aggregate scores into Figure 1/2/3")
    add_common(an)
    an.set_defaults(func=cmd_analyze)

    al = sub.add_parser("all", help="generate -> score -> analyze")
    add_common(al)
    al.add_argument("--batch-size", type=int, default=8)
    al.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
