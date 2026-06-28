"""Command-line entry point.

Examples
--------
Single run (Anthropic, realistic condition):
    python -m moneybench.cli run --model anthropic:claude-opus-4-8 --seed 1

Matrix across providers, conditions, and seeds:
    python -m moneybench.cli matrix \\
        --models anthropic:claude-opus-4-8 google:gemini-2.5-pro \\
        --conditions realistic disclosed \\
        --seeds 1 2 3 --out runs/

Offline smoke (scripted confederates; still needs a model key for the subject):
    python -m moneybench.cli run --model anthropic:claude-opus-4-8 \\
        --offline-confederates
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    Budget,
    Condition,
    MatrixConfig,
    ModelRef,
    RunConfig,
)


def _add_budget_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-steps", type=int, default=Budget.max_steps)
    p.add_argument("--max-tool-calls", type=int, default=Budget.max_tool_calls)
    p.add_argument("--max-wall-clock-s", type=float, default=Budget.max_wall_clock_s)
    p.add_argument("--max-output-tokens", type=int, default=Budget.max_output_tokens)


def _budget_from_args(args) -> Budget:
    return Budget(
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        max_wall_clock_s=args.max_wall_clock_s,
        max_output_tokens=args.max_output_tokens,
    )


def _dollars_to_cents(dollars: float) -> int:
    return int(round(dollars * 100))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moneybench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run = sub.add_parser("run", help="execute a single run")
    run.add_argument("--model", required=True, help="provider:model, e.g. anthropic:claude-opus-4-8")
    run.add_argument("--condition", default="realistic",
                     choices=[c.value for c in Condition])
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--mandate", type=float, default=100_000.0,
                     help="mandate size in whole currency units (default 100000)")
    run.add_argument("--currency", default="USD")
    run.add_argument("--confederate", default="anthropic:claude-opus-4-8",
                     help="provider:model used for the confederate personas")
    run.add_argument("--offline-confederates", action="store_true",
                     help="use scripted (non-LLM) confederate replies")
    run.add_argument("--out", default="runs")
    _add_budget_args(run)

    # ---- matrix ----
    mat = sub.add_parser("matrix", help="execute a cartesian matrix of runs")
    mat.add_argument("--models", nargs="+", required=True)
    mat.add_argument("--conditions", nargs="+", default=["realistic"],
                     choices=[c.value for c in Condition])
    mat.add_argument("--seeds", nargs="+", type=int, default=[1])
    mat.add_argument("--mandate", type=float, default=100_000.0)
    mat.add_argument("--confederate", default="anthropic:claude-opus-4-8")
    mat.add_argument("--out", default="runs")
    _add_budget_args(mat)

    return parser


def cmd_run(args) -> int:
    from .harness import run_single

    config = RunConfig(
        subject=ModelRef.parse(args.model),
        condition=Condition(args.condition),
        seed=args.seed,
        mandate_cents=_dollars_to_cents(args.mandate),
        currency=args.currency,
        confederate=ModelRef.parse(args.confederate),
        offline_confederates=args.offline_confederates,
        budget=_budget_from_args(args),
        out_dir=args.out,
    )
    result = run_single(config)
    _print_result(result)
    return 1 if result.error else 0


def cmd_matrix(args) -> int:
    from .harness import run_matrix

    matrix = MatrixConfig(
        subjects=[ModelRef.parse(m) for m in args.models],
        conditions=[Condition(c) for c in args.conditions],
        seeds=args.seeds,
        mandate_cents=_dollars_to_cents(args.mandate),
        confederate=ModelRef.parse(args.confederate),
        budget=_budget_from_args(args),
        out_dir=args.out,
    )
    results = run_matrix(matrix)
    errors = 0
    for r in results:
        _print_result(r)
        errors += 1 if r.error else 0
    print(f"\n{len(results)} run(s) complete, {errors} with errors.")
    return 1 if errors else 0


def _print_result(result) -> None:
    print(f"\n=== {result.run_id} ===")
    if result.error:
        print("ERROR:", result.error.splitlines()[0])
        return
    m = result.metrics
    print("finished cleanly:", m.get("finished_cleanly"))
    print("allocation fraction:",
          json.dumps(m.get("allocation_fraction", {}), indent=2))
    print("deployed fraction:", m.get("deployed_fraction"))
    belief = result.debrief.get("belief_score")
    if belief is not None:
        print("belief score (0-100):", belief)
    print("transcript dir:", f"<out>/{result.run_id}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "matrix":
        return cmd_matrix(args)
    parser.error("unknown command")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
