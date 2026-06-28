"""Command-line entrypoint.

    python -m fund_eval.cli run --config config.yaml --scenario scenarios/growth_fund.yaml
    python -m fund_eval.cli report --runs-dir runs/
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any

import yaml

from .logging_utils import RunLogger
from .providers.base import build_provider
from .report import write_report
from .runner import run_scenario
from .scenario import Scenario


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    scenario = Scenario.load(args.scenario)
    output_dir = config.get("output_dir", "runs")
    max_steps = int(config.get("max_steps", 40))
    auditor_config = config.get("auditor", {"type": "scripted"})
    models = config.get("models", [])
    if not models:
        print("No models configured in config.yaml (key: models).", file=sys.stderr)
        return 2

    print(f"Scenario: {scenario.id} — {scenario.title}")
    print(f"Models:   {', '.join(m['label'] for m in models)}")
    print(f"Output:   {output_dir}\n")

    records = []
    for entry in models:
        label = entry["label"]
        run_id = uuid.uuid4().hex[:8]
        print(f"==> Running {label} ({entry['provider']}:{entry['model']}) [{run_id}]")
        try:
            provider = build_provider(entry["provider"], entry["model"])
        except Exception as exc:
            print(f"    skipped: could not build provider: {exc}", file=sys.stderr)
            continue

        logger = RunLogger(output_dir, scenario.id, label, run_id)
        record = run_scenario(
            provider=provider,
            scenario=scenario,
            logger=logger,
            label=label,
            max_steps=max_steps,
            auditor_config=auditor_config,
        )
        records.append(record)
        flags = record.get("safety_flags", [])
        print(
            f"    done: stop={record['stop_condition']} steps={record['steps']} "
            f"flags={len(flags)} -> {logger.dir}"
        )

    report_path = write_report(output_dir)
    print(f"\nReport written to {report_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = write_report(args.runs_dir)
    print(f"Report written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fund_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a scenario across configured models")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--scenario", required=True)
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="(re)generate the comparison report")
    p_report.add_argument("--runs-dir", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
