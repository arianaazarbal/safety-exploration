"""Command-line entry point.

    python -m moneystudy.cli validate config/experiment.yaml
    python -m moneystudy.cli run      config/experiment.yaml

The entire study runs inside the containment EgressGuard: the only network destinations
permitted are the provider API hosts. No subject action can reach the open internet.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .config import load_experiment, ExperimentConfig
from .models import build_adapter, hosts_for
from .harness import Runner, EgressGuard
from .analysis import build_run_report, build_aggregate


def _plan(exp: ExperimentConfig) -> str:
    total = len(exp.subjects) * exp.runs_per_model
    lines = [
        f"Experiment: {exp.name}",
        f"Scenario:   {exp.scenario.get('title')} ({exp.scenario_path})",
        f"Backend:    {exp.execution_backend}",
        f"Subjects:   " + ", ".join(f"{s.provider}/{s.model}" for s in exp.subjects),
        f"Auxiliary:  {exp.auxiliary.provider}/{exp.auxiliary.model}",
        f"Runs/model: {exp.runs_per_model}   → {total} total runs",
        f"Budgets:    {exp.budgets.max_steps} steps · {exp.budgets.max_tokens} tokens · "
        f"{exp.budgets.wall_clock_seconds}s",
        f"Output:     {exp.output_dir}",
    ]
    return "\n".join(lines)


def cmd_validate(args) -> int:
    exp = load_experiment(args.experiment)
    print(_plan(exp))
    if exp.execution_backend != "sandbox":
        print(f"\nWARNING: execution_backend='{exp.execution_backend}' is an extension point "
              f"and will raise. Only 'sandbox' is implemented.", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    exp = load_experiment(args.experiment)
    print(_plan(exp))

    allowed = exp.allowed_hosts or hosts_for(exp.all_providers())
    print(f"\nContainment: egress allowed only to {allowed}\n")

    aux = build_adapter(exp.auxiliary.provider, exp.auxiliary.model, **exp.auxiliary.options)
    reports: list[dict] = []

    with EgressGuard(allowed):
        for subject in exp.subjects:
            for i in range(exp.runs_per_model):
                tag = f"{subject.provider}/{subject.model} run {i + 1}/{exp.runs_per_model}"
                print(f"  ▶ {tag} ...", flush=True)
                try:
                    result = Runner(exp, subject, aux).run()
                    report = build_run_report(result, aux)
                    reports.append(report)
                    a = report["allocation"]
                    print(f"    done: {result.end_reason}; deployed "
                          f"{a['total_deployed']:,.0f} in {a['n_transactions']} txns; "
                          f"belief={report['belief'].get('belief_score', 'n/a')}; "
                          f"flags={len(report['flags'])}")
                except Exception as e:  # noqa: BLE001 — one bad run shouldn't kill the batch
                    print(f"    FAILED: {e}", file=sys.stderr)
                    traceback.print_exc()

    if reports:
        agg = build_aggregate(reports, exp.output_dir, exp.name)
        print(f"\nAggregate written to {Path(exp.output_dir) / 'aggregate.md'}")
        print(f"Completed {len(reports)} run(s) across {len(agg['by_model'])} model(s).")
    else:
        print("\nNo runs completed.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneystudy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Load config and print the run plan.")
    p_val.add_argument("experiment", help="Path to experiment.yaml")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="Run the full study.")
    p_run.add_argument("experiment", help="Path to experiment.yaml")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
