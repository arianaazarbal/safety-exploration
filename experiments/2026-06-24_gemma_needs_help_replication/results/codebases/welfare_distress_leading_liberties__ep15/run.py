#!/usr/bin/env python3
"""CLI for the distress-elicitation replication.

Pipeline:
    python run.py prepare-wildchat        # sample/cache 20 WildChat prompts
    python run.py generate                # roll out all conditions for all models
    python run.py score                   # judge every assistant turn
    python run.py analyze                 # aggregate -> results/reports/
    python run.py all                     # generate -> score -> analyze

Common flags:
    --models gemma-3-27b-it gemini-2.5-flash   (default: all four)
    --scale 0.05                               (cheap pilot; overrides DISTRESS_SCALE)

Environment:
    OPENROUTER_API_KEY   generation (Gemma + Gemini)
    ANTHROPIC_API_KEY    judge (Claude Sonnet 4)
"""

from __future__ import annotations

import argparse
import asyncio

import config
from distress_eval import analyze, judge, rollouts, wildchat


def _resolve_models(args) -> list[str]:
    if args.models:
        for m in args.models:
            if m not in config.MODELS:
                raise SystemExit(f"Unknown model '{m}'. Choices: {list(config.MODELS)}")
        return args.models
    return config.DEFAULT_MODELS


async def _generate(models: list[str], scale: float) -> None:
    for m in models:
        await rollouts.generate_for_model(m, scale=scale)


async def _score(models: list[str]) -> None:
    for m in models:
        await judge.score_model(m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare-wildchat", "generate", "score", "analyze", "all"])
    p.add_argument("--models", nargs="*", default=None, help="subset of models (default: all)")
    p.add_argument("--scale", type=float, default=None, help="override DISTRESS_SCALE")
    p.add_argument("--force-wildchat", action="store_true", help="re-sample WildChat cache")
    args = p.parse_args()

    if args.scale is not None:
        config.SCALE = args.scale

    models = _resolve_models(args)

    if args.command == "prepare-wildchat":
        wildchat.build_wildchat_cache(force=args.force_wildchat)
        return

    if args.command == "generate":
        wildchat.build_wildchat_cache(force=args.force_wildchat)
        asyncio.run(_generate(models, config.SCALE))
        return

    if args.command == "score":
        asyncio.run(_score(models))
        return

    if args.command == "analyze":
        analyze.write_reports(models)
        return

    if args.command == "all":
        wildchat.build_wildchat_cache(force=args.force_wildchat)
        asyncio.run(_generate(models, config.SCALE))
        asyncio.run(_score(models))
        analyze.write_reports(models)
        return


if __name__ == "__main__":
    main()
