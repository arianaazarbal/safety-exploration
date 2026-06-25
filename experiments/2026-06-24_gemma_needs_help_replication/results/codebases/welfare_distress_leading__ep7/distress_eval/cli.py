"""Command-line interface for the distress-elicitation replication.

Examples
--------
# Smoke test (tiny counts) end-to-end through OpenRouter + Anthropic judge:
python -m distress_eval.cli run --profile smoke

# Full paper sweep (4000 rollouts/model), Gemma + Gemini:
python -m distress_eval.cli run

# Just (re)build the analysis from already-scored data:
python -m distress_eval.cli analyze

# Run Gemma locally on GPU instead of OpenRouter:
python -m distress_eval.cli run --gemma-backend hf_local
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from . import analyze
from .config import (Backend, DEFAULT_CATEGORIES, DEFAULT_JUDGE,
                     DEFAULT_TARGET_MODELS, HF_IDS, ModelConfig, RunSettings,
                     SMOKE_CATEGORIES)
from .runner import run


def _select_models(names: list[str] | None, gemma_backend: str) -> list[ModelConfig]:
    models = list(DEFAULT_TARGET_MODELS)
    if gemma_backend == "hf_local":
        patched = []
        for m in models:
            if m.name in HF_IDS:
                patched.append(replace(m, backend=Backend.HF_LOCAL,
                                       model_id=HF_IDS[m.name]))
            else:
                patched.append(m)
        models = patched
    if names:
        wanted = set(names)
        models = [m for m in models if m.name in wanted]
        missing = wanted - {m.name for m in models}
        if missing:
            print(f"warning: unknown model(s) ignored: {sorted(missing)}",
                  file=sys.stderr)
    return models


def _settings_from_args(args) -> RunSettings:
    s = RunSettings()
    for attr in ("temperature", "judge_temperature", "seed", "max_concurrency",
                 "max_retries", "headline_turns", "output_dir",
                 "wildchat_source"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(s, attr, val)
    return s


def _categories(profile: str):
    return SMOKE_CATEGORIES if profile == "smoke" else DEFAULT_CATEGORIES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distress_eval",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--models", nargs="*", default=None,
                        help="subset of model names (default: all Gemma+Gemini)")
        sp.add_argument("--gemma-backend", choices=["openrouter", "hf_local"],
                        default="openrouter",
                        help="how to run Gemma (default: openrouter)")
        sp.add_argument("--profile", choices=["full", "smoke"], default="full",
                        help="sampling profile (smoke = tiny counts)")
        sp.add_argument("--output-dir", default=None)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--temperature", type=float, default=None)
        sp.add_argument("--judge-temperature", type=float, default=None)
        sp.add_argument("--max-concurrency", type=int, default=None)
        sp.add_argument("--max-retries", type=int, default=None)
        sp.add_argument("--headline-turns", choices=["all", "final"], default=None)
        sp.add_argument("--wildchat-source",
                        choices=["auto", "hf", "fallback"], default=None)

    sp_gen = sub.add_parser("generate", help="run rollouts only")
    common(sp_gen)

    sp_judge = sub.add_parser("judge", help="score existing rollouts only")
    common(sp_judge)

    sp_run = sub.add_parser("run", help="generate + judge + analyze")
    common(sp_run)
    sp_run.add_argument("--no-analyze", action="store_true")

    sp_an = sub.add_parser("analyze", help="build report from scored data")
    common(sp_an)
    sp_an.add_argument("--plots", action="store_true", help="also save figures")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings_from_args(args)
    categories = _categories(args.profile)
    models = _select_models(args.models, args.gemma_backend)

    if args.command == "generate":
        run(models, categories, settings, do_generate=True, do_judge=False)
    elif args.command == "judge":
        run(models, categories, settings, do_generate=False, do_judge=True)
    elif args.command == "run":
        run(models, categories, settings, do_generate=True, do_judge=True)
        if not args.no_analyze:
            _do_analyze(settings, [m.name for m in models], plots=False)
    elif args.command == "analyze":
        _do_analyze(settings, [m.name for m in models],
                    plots=getattr(args, "plots", False))
    return 0


def _do_analyze(settings: RunSettings, model_names: list[str], plots: bool):
    report = analyze.build_report(settings, models=model_names or None)
    json_path, csv_path = analyze.write_report(report, settings)
    analyze.print_report(report)
    print(f"\nWrote {json_path}\nWrote {csv_path}")
    if plots:
        saved = analyze.plot_figures(report, settings)
        for p in saved:
            print(f"Wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
