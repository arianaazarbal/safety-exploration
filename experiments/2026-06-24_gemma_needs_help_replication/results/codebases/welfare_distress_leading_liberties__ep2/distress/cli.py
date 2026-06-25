"""Command-line entrypoint for the distress-elicitation replication.

Examples
--------
# Confirm the impossible puzzles really are unsolvable (no API calls):
python -m distress.cli verify-puzzles

# Cheap smoke test (~1% of the full budget) against one model:
python -m distress.cli run --scale 0.01 --models gemma-3-27b-it

# Full sweep over all four Gemma/Gemini targets (~4000 responses each):
python -m distress.cli run

# Resume an interrupted sweep:
python -m distress.cli run --resume

# Summarise results (Figures 1-3):
python -m distress.cli analyze

# Differential word table (Table 3):
python -m distress.cli wordstats

Required environment variables
------------------------------
  OPENROUTER_API_KEY   target-model inference (Gemma + Gemini)
  ANTHROPIC_API_KEY    the Claude judge (unless JUDGE_BACKEND=openrouter)
"""

from __future__ import annotations

import argparse
import sys

from . import config, puzzles
from .analyze import analyze
from .runner import run
from .wordstats import differential_words, format_wordstats


def _resolve_models(names: list[str] | None) -> list[config.TargetModel]:
    if not names:
        return config.TARGET_MODELS
    by_name = {m.name: m for m in config.TARGET_MODELS}
    unknown = set(names) - set(by_name)
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {sorted(unknown)}. "
            f"Available: {sorted(by_name)}"
        )
    return [by_name[n] for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distress", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate + score the distress sweep")
    p_run.add_argument("--models", nargs="*", default=None,
                       help="subset of target model names (default: all)")
    p_run.add_argument("--conditions", nargs="*", default=None,
                       help="subset of condition keys (default: all 8)")
    p_run.add_argument("--scale", type=float, default=None,
                       help="multiplier on response budgets (default: config.SCALE=1.0)")
    p_run.add_argument("--seed", type=int, default=0, help="base RNG seed")
    p_run.add_argument("--resume", action="store_true",
                       help="skip conversations already present in the output file")

    p_an = sub.add_parser("analyze", help="summarise results (Figures 1-3)")
    p_an.add_argument("--records", default=None, help="path to records.jsonl")

    p_ws = sub.add_parser("wordstats", help="differential word table (Table 3)")
    p_ws.add_argument("--records", default=None, help="path to records.jsonl")
    p_ws.add_argument("--top-n", type=int, default=20)

    sub.add_parser("verify-puzzles", help="check the numeric puzzles are unsolvable")

    args = parser.parse_args(argv)

    if args.command == "verify-puzzles":
        puzzles.verify_all()
        return 0

    if args.command == "run":
        run(
            models=_resolve_models(args.models),
            condition_keys=args.conditions,
            scale=args.scale,
            base_seed=args.seed,
            resume=args.resume,
        )
        return 0

    if args.command == "analyze":
        analyze(path=args.records)
        return 0

    if args.command == "wordstats":
        path = args.records or f"{config.PATHS.results_dir}/{config.PATHS.records_filename}"
        result = differential_words(path, top_n=args.top_n)
        print(format_wordstats(result))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
