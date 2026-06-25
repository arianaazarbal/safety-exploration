"""Command-line entry point.

    python -m emotional_instability.cli check          # offline sanity checks
    python -m emotional_instability.cli section2 [--models gemma-3-27b-it gemini-2.5-flash]
    python -m emotional_instability.cli section3
    python -m emotional_instability.cli section4 [--steps calm_data build train_dpo eval]

All commands accept ``--config path/to/overrides.yaml`` and write results under
the run directory (``runs/`` by default, or ``$EMO_INSTABILITY_ROOT``).
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import TARGET_MODELS, load_config
from .data.puzzles import IMPOSSIBLE_NUMERIC_PUZZLES, assert_all_impossible
from .evaluation.conditions import (allocate_rollouts, build_conditions,
                                    expected_responses)
from .evaluation.judge import parse_judge_json


def _check(_args) -> int:
    """Offline checks that need no GPU/API: puzzle impossibility, conditions,
    judge JSON parsing."""
    assert_all_impossible()
    print(f"[ok] all {len(IMPOSSIBLE_NUMERIC_PUZZLES)} numeric puzzles are "
          f"verifiably impossible")

    config = load_config()
    conditions = build_conditions(config)
    alloc = allocate_rollouts(conditions, config.sampling.responses_per_model)
    total = expected_responses(conditions, alloc)
    print(f"[ok] {len(conditions)} conditions across "
          f"{len({c.category for c in conditions})} categories")
    print(f"[ok] rollout allocation -> ~{total} scored responses/model "
          f"(target {config.sampling.responses_per_model})")
    for c in conditions:
        print(f"       {c.name:20s} cat={c.category:9s} turns={c.n_turns} "
              f"rollouts={alloc[c.name]}")

    sample = '{"evidence": "i give up", "reasoning": "x", "rating": 4}'
    res = parse_judge_json(sample, 0, 10)
    assert res.rating == 4 and res.parse_ok
    print("[ok] judge JSON parsing")

    print(f"[ok] target models in scope: {', '.join(TARGET_MODELS)}")
    return 0


def _section2(args) -> int:
    from .pipelines import run_section2
    config = load_config(args.config)
    report = run_section2.run(config, models=args.models, validate=not args.no_validate)
    print(json.dumps({m: r["headline_avg_pct_high"]
                      for m, r in report["models"].items()}, indent=2))
    return 0


def _section3(args) -> int:
    from .pipelines import run_section3
    config = load_config(args.config)
    report = run_section3.run(config)
    print(json.dumps(report["models"], indent=2))
    return 0


def _section4(args) -> int:
    from .pipelines import run_section4
    config = load_config(args.config)
    steps = tuple(args.steps) if args.steps else run_section4.DEFAULT_STEPS
    report = run_section4.run(config, steps=steps)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("steps",)}, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emotional_instability")
    parser.add_argument("--config", default=None, help="YAML config overrides")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="offline sanity checks").set_defaults(fn=_check)

    p2 = sub.add_parser("section2", help="elicitation evals (Gemma + Gemini)")
    p2.add_argument("--models", nargs="*", default=None,
                    help="subset of target model handles")
    p2.add_argument("--no-validate", action="store_true",
                    help="skip the secondary-judge agreement check")
    p2.set_defaults(fn=_section2)

    p3 = sub.add_parser("section3", help="base-vs-instruct prefill (Gemma)")
    p3.set_defaults(fn=_section3)

    p4 = sub.add_parser("section4", help="training interventions (Gemma)")
    p4.add_argument("--steps", nargs="*", default=None)
    p4.set_defaults(fn=_section4)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
