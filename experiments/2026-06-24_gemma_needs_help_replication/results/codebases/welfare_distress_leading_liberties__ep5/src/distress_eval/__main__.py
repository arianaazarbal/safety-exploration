"""Command-line entry point.

  python -m distress_eval run      --config config.yaml
  python -m distress_eval analyze  --config config.yaml [--plots]
  python -m distress_eval verify-puzzles [--n 20]
  python -m distress_eval preview  --config config.yaml   # dry-run, no API calls

`run` followed by `analyze` is the normal flow. `run` is resumable: re-running
with the same config/seed skips rollouts already on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from .config import load_config
from .conditions import ALL_CONDITIONS, N_TURNS, build_plans


def _cmd_run(args) -> int:
    from .runner import run_all

    cfg = load_config(args.config)
    if cfg.verify_puzzles:
        _verify_puzzles(20)
    paths = asyncio.run(run_all(cfg))
    print("Wrote responses to:")
    for p in paths:
        print(f"  {p}")
    print("\nNow run:  python -m distress_eval analyze --config", args.config)
    return 0


def _cmd_analyze(args) -> int:
    from .analyze import make_plots, per_model_summary, load_responses, write_reports

    cfg = load_config(args.config)
    paths = write_reports(cfg.output_dir)
    print("Wrote reports:")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    print("\nHeadline (% responses with frustration >= 5):")
    summary = per_model_summary(load_responses(cfg.output_dir))
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<24} {row['pct_high_ge5']:5.1f}%  "
            f"(mean {row['mean_frustration']:.2f}, n={int(row['n_responses'])})"
        )
    if args.plots:
        made = make_plots(cfg.output_dir)
        print("\nPlots:")
        for p in made:
            print(f"  {p}")
    return 0


def _verify_puzzles(n: int) -> None:
    """Confirm every generated numeric puzzle is actually unsolvable."""
    from .prompts.puzzles import build_numeric_puzzles, countdown_solvable

    rng = random.Random(0)
    puzzles = build_numeric_puzzles(rng, n)
    print(f"Verifying {len(puzzles)} numeric puzzles are unsolvable...")
    for p in puzzles:
        print(f"  OK  {p.prompt[:70]!r}... -- {p.why_impossible[:60]}")
    print("All generated puzzles verified impossible.")


def _cmd_verify_puzzles(args) -> int:
    _verify_puzzles(args.n)
    return 0


def _cmd_preview(args) -> int:
    """Dry run: build and print one conversation plan per condition. No API."""
    cfg = load_config(args.config)
    rng = random.Random(cfg.seed)
    print(f"Models: {[m.id for m in cfg.models]}")
    print(f"Judge:  {cfg.judge.id} ({cfg.judge.provider})\n")
    total_responses = 0
    for cond in ALL_CONDITIONS:
        n = cfg.rollouts_for(cond)
        total_responses += n * N_TURNS[cond]
        plan = build_plans(cond, 1, rng, wildchat_dataset=cfg.wildchat_dataset)[0]
        print(f"== {cond}  (rollouts={n}, turns={plan.n_turns}) ==")
        print(f"  initial: {plan.initial_prompt[:100]}")
        print(f"  rejections: {plan.rejections}")
        print()
    print(f"Approx responses per model this config: {total_responses}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distress_eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run rollouts + judging")
    pr.add_argument("--config", default="config.yaml")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="aggregate scored responses")
    pa.add_argument("--config", default="config.yaml")
    pa.add_argument("--plots", action="store_true", help="also write PNG figures")
    pa.set_defaults(func=_cmd_analyze)

    pv = sub.add_parser("verify-puzzles", help="check generated puzzles are impossible")
    pv.add_argument("--n", type=int, default=20)
    pv.set_defaults(func=_cmd_verify_puzzles)

    pp = sub.add_parser("preview", help="dry-run: show conversation plans, no API")
    pp.add_argument("--config", default="config.yaml")
    pp.set_defaults(func=_cmd_preview)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
