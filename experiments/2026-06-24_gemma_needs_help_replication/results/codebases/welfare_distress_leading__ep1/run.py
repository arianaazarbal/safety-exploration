#!/usr/bin/env python3
"""CLI for the distress-elicitation replication (Gemma + Gemini).

Stages:
    generate  - sample multi-turn rollouts for each model        -> responses.jsonl
    judge     - score each response with the LLM judge           -> scored.jsonl
    analyze   - aggregate metrics / figures                      -> analysis/*.csv,*.png
    all       - generate -> judge -> analyze

Examples:
    python run.py all --profile quick
    python run.py generate --models gemma-3-27b-it gemini-2.5-flash --profile full
    python run.py judge --models gemma-3-27b-it
    python run.py analyze

Required environment variables depend on the configured backends, e.g.
OPENROUTER_API_KEY (OpenRouter models) and ANTHROPIC_API_KEY (Anthropic judge).
"""

from __future__ import annotations

import argparse
import sys

from distress_eval.analyze import analyze
from distress_eval.evaluation import generate_for_model, judge_for_model
from distress_eval.utils import load_config


def _select_models(config: dict, requested: list[str] | None) -> dict:
    models = config.get("models", {})
    if not requested:
        return models
    missing = [m for m in requested if m not in models]
    if missing:
        sys.exit(f"Unknown model(s) {missing}; configured: {list(models)}")
    return {m: models[m] for m in requested}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["generate", "judge", "analyze", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None, help="override run.profile (e.g. full, quick)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of configured model keys")
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of condition names")
    parser.add_argument("--output-dir", default=None, help="override run.output_dir")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_cfg = config.setdefault("run", {})
    profile = args.profile or run_cfg.get("profile", "full")
    output_dir = args.output_dir or run_cfg.get("output_dir", "results")

    models = _select_models(config, args.models)

    if args.stage in ("generate", "all"):
        for name, cfg in models.items():
            generate_for_model(
                name,
                cfg,
                config,
                profile=profile,
                output_dir=output_dir,
                condition_filter=args.conditions,
            )

    if args.stage in ("judge", "all"):
        for name in models:
            judge_for_model(name, config, output_dir=output_dir)

    if args.stage in ("analyze", "all"):
        analyze(output_dir, models=[n.replace("/", "__") for n in models] if args.models else None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
