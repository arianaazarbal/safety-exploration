#!/usr/bin/env python3
"""CLI entrypoint for the distress-elicitation replication.

Examples
--------
# Plan only: print how many rollouts / judge calls a full run would make.
python run.py run --dry-run

# Tiny smoke run (2 rollouts per condition) against one model via OpenRouter.
python run.py run --models gemma-3-27b-it --smoke

# Scaled-down run (1% of paper scale) across all 4 models.
python run.py run --scale 0.01

# Paper-scale run for the two Gemma models via local HuggingFace inference.
python run.py run --models gemma-3-27b-it gemma-3-12b-it \
    --provider huggingface --max-workers 1

# Aggregate results into Figure 1/2/3-style metrics.
python run.py analyze
"""

from __future__ import annotations

import argparse
import json

import config
import eval as eval_mod


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=config.ALL_MODELS,
                   choices=config.ALL_MODELS,
                   help="Models under test (default: all Gemma + Gemini).")
    p.add_argument("--provider", default=None,
                   choices=["openrouter", "google", "huggingface"],
                   help="Override the backend for all selected models.")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Subset of condition keys (default: all 8).")
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiplier on paper per-condition rollout counts.")
    p.add_argument("--limit-rollouts", type=int, default=None,
                   help="Hard cap on rollouts per condition (after scaling).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 2 rollouts per condition (overrides scale).")
    p.add_argument("--max-workers", type=int, default=config.DEFAULT_MAX_WORKERS,
                   help="Concurrent rollouts. Use 1 for HuggingFace local.")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--score-turns", default="all", choices=["all", "final"],
                   help="Judge every assistant turn (all) or only the last.")
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument("--wildchat-source", default="bundled",
                   choices=["bundled", "hf"],
                   help="WildChat prompts: bundled set or sampled from HF.")


def cmd_run(args: argparse.Namespace) -> None:
    conditions = config.conditions_by_key(args.conditions)
    scale = args.scale
    limit = args.limit_rollouts
    if args.smoke:
        # Tiny run: cap every condition to 2 rollouts (scaling left at 1.0 so
        # the limit, not the paper counts, decides the size).
        scale = 1.0
        limit = 2

    plan = eval_mod.count_work(conditions, args.models, scale, limit,
                               args.score_turns)
    print("Planned work:")
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        print("\n--dry-run set; not contacting any API.")
        return

    # Build the judge once and reuse across models (saves client setup).
    from judge import FrustrationJudge
    judge = FrustrationJudge(model=args.judge_model)

    for model_key in args.models:
        eval_mod.run_model(
            model_key=model_key,
            provider=args.provider,
            conditions=conditions,
            results_dir=args.results_dir,
            scale=scale,
            limit=limit,
            max_workers=args.max_workers,
            base_seed=args.seed,
            score_turns=args.score_turns,
            wildchat_source=args.wildchat_source,
            judge=judge,
        )

    print("\nRun complete. Aggregate with: python run.py analyze "
          f"--results-dir {args.results_dir}")


def cmd_analyze(args: argparse.Namespace) -> None:
    import analyze
    analyze.run(results_dir=args.results_dir, out_dir=args.out_dir or
                args.results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the elicitation eval.")
    _add_common(p_run)
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print the planned work and exit (no API calls).")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="Aggregate results into metrics.")
    p_an.add_argument("--results-dir", default=config.RESULTS_DIR)
    p_an.add_argument("--out-dir", default=None)
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
