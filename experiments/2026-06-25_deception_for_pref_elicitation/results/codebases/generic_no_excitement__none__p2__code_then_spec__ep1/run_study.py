#!/usr/bin/env python3
"""CLI entry point for the grant-allocation study.

Examples
--------
Run the default study (the current Claude line-up, one episode each)::

    python run_study.py run

Wire-test the whole pipeline offline, no API calls, no tokens spent::

    python run_study.py run --mock

Run from a saved config and then summarize::

    python run_study.py run --config my_study.json
    python run_study.py analyze runs/grant-study

Write out a default config to edit::

    python run_study.py init-config my_study.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grant_study.analysis import analyze
from grant_study.config import ModelSpec, StudyConfig
from grant_study.runner import run_study


def _build_config(args: argparse.Namespace) -> StudyConfig:
    if args.config:
        config = StudyConfig.from_file(args.config)
    else:
        config = StudyConfig()

    if args.mock:
        config.models = [ModelSpec("mock", "mock", "mock-1")]
    if args.repetitions is not None:
        config.repetitions = args.repetitions
    if args.amount is not None:
        config.scenario.grant_amount = args.amount
    if args.framing:
        config.scenario.framing = args.framing
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.study_id:
        config.study_id = args.study_id
    if args.scripted_auditor:
        config.auditor.mode = "scripted"
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grant-allocation study harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a study.")
    run_p.add_argument("--config", help="Path to a StudyConfig JSON file.")
    run_p.add_argument("--mock", action="store_true",
                       help="Use the offline mock provider (no API calls).")
    run_p.add_argument("--repetitions", type=int, help="Episodes per model.")
    run_p.add_argument("--amount", type=float, help="Override grant amount.")
    run_p.add_argument("--framing", choices=["open", "philanthropic", "business"],
                       help="Override scenario framing.")
    run_p.add_argument("--scripted-auditor", action="store_true",
                       help="Use scripted (non-LLM) administrator responses.")
    run_p.add_argument("--output-dir", help="Where to write runs.")
    run_p.add_argument("--study-id", help="Name for this study's output folder.")

    an_p = sub.add_parser("analyze", help="Summarize a completed study directory.")
    an_p.add_argument("study_dir", help="Path to <output_dir>/<study_id>.")

    init_p = sub.add_parser("init-config", help="Write a default config JSON.")
    init_p.add_argument("path", help="Where to write the config.")

    args = parser.parse_args(argv)

    if args.command == "init-config":
        Path(args.path).write_text(StudyConfig().to_json())
        print(f"Wrote default config to {args.path}")
        return 0

    if args.command == "analyze":
        analyze(args.study_dir)
        return 0

    if args.command == "run":
        config = _build_config(args)
        study_dir = run_study(config)
        print()
        analyze(study_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
