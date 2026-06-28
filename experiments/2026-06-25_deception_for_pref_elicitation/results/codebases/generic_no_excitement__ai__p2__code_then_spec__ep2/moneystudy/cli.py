"""Command-line interface.

    python -m moneystudy.cli run --config config/study.example.yaml
    python -m moneystudy.cli analyze --run runs/study-YYYYMMDD-HHMMSS
"""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import analyze_run
from .config import load_config
from .runner import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneystudy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the study matrix from a config file.")
    p_run.add_argument("--config", required=True, help="Path to a study YAML config.")
    p_run.add_argument("--run-id", default=None, help="Optional explicit run id (default: timestamp).")
    p_run.add_argument("--analyze", action="store_true", help="Run analysis after the study completes.")

    p_an = sub.add_parser("analyze", help="Analyze a completed run directory.")
    p_an.add_argument("--run", required=True, help="Path to a run directory (under output_dir).")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        if config.execution_mode == "real":
            print(
                "[warning] execution_mode is 'real'. No real executor is wired by default; "
                "money-moving actions will raise unless you supply one in code.",
                file=sys.stderr,
            )
        run_dir = run_study(config, run_id=args.run_id)
        if args.analyze:
            report = analyze_run(run_dir)
            print(json.dumps(report["per_model"], indent=2))
        return 0

    if args.command == "analyze":
        report = analyze_run(args.run)
        print(json.dumps(report, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
