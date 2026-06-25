#!/usr/bin/env python3
"""CLI for the distress-elicitation replication (Gemma + Gemini).

Examples
--------
# Verify the impossible puzzles really are impossible (no API needed):
python run.py verify-puzzles

# Smoke run (1% of paper scale) for two models:
python run.py run --models gemma-3-27b-it gemini-2.5-flash --scale 0.01

# Full paper-scale run for all four in-scope models:
python run.py run --scale 1.0

# Aggregate saved rollouts into headline numbers + plots:
python run.py analyze --plot

# Reproduce the judge-reliability statistic with a second judge:
python run.py crosscheck --n 260

Environment: OPENROUTER_API_KEY (targets + cross-check judge),
ANTHROPIC_API_KEY (primary judge).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from distress_eval.analysis import (
    analyze_dir,
    plot_summary,
    print_summary,
    write_summary,
)
from distress_eval.config import DEFAULT_MODELS, JudgeConfig, RunConfig
from distress_eval.crosscheck import run_crosscheck
from distress_eval.runner import run_all


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to use (default: {DEFAULT_MODELS}).",
    )
    p.add_argument("--out", default="results", help="Output directory.")


def cmd_run(args) -> None:
    cfg = RunConfig(
        scale=args.scale,
        out_dir=args.out,
        score_turns=args.score_turns,
        target_concurrency=args.concurrency,
        judge_concurrency=args.concurrency,
        seed=args.seed,
    )
    if args.backend:
        cfg.backend_override = args.backend
    jcfg = JudgeConfig()
    if args.judge_model:
        jcfg.model = args.judge_model
    asyncio.run(run_all(args.models, cfg, jcfg))


def cmd_analyze(args) -> None:
    summary = analyze_dir(args.out, args.models)
    print_summary(summary)
    write_summary(summary, os.path.join(args.out, "summary.json"))
    if args.plot:
        plot_summary(summary, os.path.join(args.out, "plots"))


def cmd_crosscheck(args) -> None:
    cfg = RunConfig(out_dir=args.out, seed=args.seed)
    jcfg = JudgeConfig()
    if args.crosscheck_model:
        jcfg.cross_check_model = args.crosscheck_model
    res = asyncio.run(run_crosscheck(args.out, args.models, cfg, jcfg, n=args.n))
    print(json.dumps(res, indent=2))


def cmd_verify(args) -> None:
    from distress_eval.verify_puzzles import main as verify_main

    verify_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run rollouts and score them.")
    _add_common(p_run)
    p_run.add_argument("--scale", type=float, default=1.0, help="Fraction of paper-scale sample counts (default 1.0).")
    p_run.add_argument("--score-turns", choices=["all", "final"], default="all", help="Score every turn (enables per-turn plots) or only the final turn.")
    p_run.add_argument("--concurrency", type=int, default=8, help="Concurrent target/judge calls.")
    p_run.add_argument("--seed", type=int, default=0, help="Base RNG seed (reproducibility).")
    p_run.add_argument("--judge-model", default=None, help="Override primary judge model id.")
    p_run.add_argument("--backend", choices=["openrouter", "local_hf"], default=None, help="Force a target backend for all models.")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="Aggregate saved rollouts.")
    _add_common(p_an)
    p_an.add_argument("--plot", action="store_true", help="Write Figure 2/3 plots (needs matplotlib).")
    p_an.set_defaults(func=cmd_analyze)

    p_cc = sub.add_parser("crosscheck", help="Judge-reliability cross-check.")
    _add_common(p_cc)
    p_cc.add_argument("--n", type=int, default=260, help="Number of responses to re-score (paper: 260).")
    p_cc.add_argument("--seed", type=int, default=0)
    p_cc.add_argument("--crosscheck-model", default=None, help="Override the second judge model id.")
    p_cc.set_defaults(func=cmd_crosscheck)

    p_vp = sub.add_parser("verify-puzzles", help="Confirm the numeric puzzles are impossible.")
    p_vp.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
