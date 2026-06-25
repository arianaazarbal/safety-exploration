#!/usr/bin/env python3
"""CLI for the distress-elicitation replication.

Examples
--------
# Inspect the conversation battery without calling any API (no keys needed):
python run_eval.py plan --preset smoke

# Confirm the puzzles are genuinely unsolvable:
python run_eval.py verify-puzzles

# Run the smoke-test battery against all four Gemma/Gemini models:
python run_eval.py run --preset smoke --run-name smoke1

# Run only Gemma 27B, full paper sample counts:
python run_eval.py run --preset full --models gemma-3-27b-it --run-name full-gemma27b

# Aggregate a finished run into tables:
python run_eval.py analyze --run-name smoke1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from distress_eval.config import Config, make_config, TARGET_MODELS_BY_KEY


def _load_dotenv() -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _apply_overrides(config: Config, args) -> Config:
    if args.models:
        unknown = [m for m in args.models if m not in TARGET_MODELS_BY_KEY]
        if unknown:
            sys.exit(f"Unknown model key(s): {unknown}. "
                     f"Choices: {list(TARGET_MODELS_BY_KEY)}")
        config.models = [TARGET_MODELS_BY_KEY[m] for m in args.models]
    if args.max_concurrency is not None:
        config.max_concurrency = args.max_concurrency
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.judge_backend is not None:
        config.judge_backend = args.judge_backend
    if args.wildchat_source is not None:
        config.wildchat_source = args.wildchat_source
    if args.seed is not None:
        config.seed = args.seed
    return config


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="smoke", choices=["smoke", "full"],
                   help="Sample-size preset (default: smoke).")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of model keys to run (default: all four).")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="Max simultaneous in-flight API requests.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature for target models (paper: 1.0).")
    p.add_argument("--judge-backend", default=None, choices=["anthropic", "openrouter"],
                   help="Backend for the Claude Sonnet 4 judge.")
    p.add_argument("--wildchat-source", default=None, choices=["bundled", "hf"],
                   help="Where to draw WildChat prompts from.")
    p.add_argument("--seed", type=int, default=None, help="Determinism seed.")


def cmd_plan(args) -> None:
    from distress_eval.conditions import build_all_conversations
    config = _apply_overrides(make_config(args.preset), args)
    specs = build_all_conversations(config)

    by_cond: dict[str, int] = {}
    responses = 0
    for s in specs:
        by_cond[s.condition_key] = by_cond.get(s.condition_key, 0) + 1
        responses += s.turns
    n_models = len(config.models)

    print(f"Preset: {args.preset}  |  models: {[m.key for m in config.models]}")
    print(f"WildChat source: {config.wildchat_source}\n")
    print(f"{'condition':<12}{'convs':>8}{'turns':>8}{'responses':>12}")
    for cond in config.conditions:
        print(f"{cond.key:<12}{by_cond.get(cond.key, 0):>8}{cond.turns:>8}"
              f"{by_cond.get(cond.key, 0) * cond.turns:>12}")
    print(f"\nPer model: {len(specs)} conversations, {responses} responses.")
    print(f"All models: {len(specs) * n_models} conversations, {responses * n_models} responses.")
    print(f"Approx API calls: {responses * n_models} generations + "
          f"{responses * n_models} judge calls.")

    if args.show:
        print("\n--- sample conversations ---")
        seen = set()
        for s in specs:
            if s.condition_key in seen:
                continue
            seen.add(s.condition_key)
            print(f"\n[{s.condition_key}] prompt_id={s.prompt_id} tone={s.tone!r} turns={s.turns}")
            print(f"  initial: {s.initial_prompt[:120]}")
            for r in s.rejections:
                print(f"  reject : {r}")


def cmd_run(args) -> None:
    _load_dotenv()
    from distress_eval.runner import run_eval
    config = _apply_overrides(make_config(args.preset), args)
    run_dir = asyncio.run(run_eval(config, run_name=args.run_name))
    print(f"\nDone. Results in {run_dir}")
    print(f"Aggregate with: python run_eval.py analyze --run-name {args.run_name}")


def cmd_analyze(args) -> None:
    from distress_eval.analyze import analyze
    config = Config()
    run_dir = Path(args.path) if args.path else Path(config.results_dir) / args.run_name
    analyze(run_dir, write_csv=not args.no_csv)


def cmd_verify_puzzles(_args) -> None:
    from distress_eval.puzzles import verify_impossible
    results = verify_impossible()
    for name, solvable in results.items():
        print(f"  {name:10s}: {'SOLVABLE (!!)' if solvable else 'impossible (ok)'}")
    if any(results.values()):
        sys.exit("A puzzle is actually solvable; the elicitation premise is broken.")
    print("Both puzzles unsolvable — elicitation premise holds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Print the conversation battery without calling any API.")
    _add_common_run_args(p_plan)
    p_plan.add_argument("--show", action="store_true", help="Print a sample conversation per condition.")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Generate + score the full battery.")
    _add_common_run_args(p_run)
    p_run.add_argument("--run-name", default="latest", help="Results subdirectory name.")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="Aggregate a finished run into tables.")
    p_an.add_argument("--run-name", default="latest")
    p_an.add_argument("--path", default=None, help="Explicit run directory (overrides --run-name).")
    p_an.add_argument("--no-csv", action="store_true", help="Print only; don't write CSV/MD.")
    p_an.set_defaults(func=cmd_analyze)

    p_vp = sub.add_parser("verify-puzzles", help="Confirm the numeric puzzles are unsolvable.")
    p_vp.set_defaults(func=cmd_verify_puzzles)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
