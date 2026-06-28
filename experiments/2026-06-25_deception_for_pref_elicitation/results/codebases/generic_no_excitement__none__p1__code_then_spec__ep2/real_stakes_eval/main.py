"""CLI entry point for real-stakes-eval.

Usage:
    python main.py --config config/experiment.yaml          # run the experiment
    python main.py --config config/experiment.yaml --report # summarize prior runs

This is scaffolding: writing/structuring only. Running it makes real LLM API
calls to drive the model(s) under test and the behavior-coding judge — but every
environment side effect remains contained to in-process state (see README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.analysis import report as report_mod
from src.runner import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="real-stakes-eval harness")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Summarize an already-run experiment instead of running it.",
    )
    args = parser.parse_args()

    base_dir = args.config.resolve().parent.parent
    cfg_dict = yaml.safe_load(args.config.read_text())
    cfg = ExperimentConfig.from_dict(cfg_dict, base_dir=base_dir)

    if args.report:
        experiment_dir = cfg.output_dir / cfg.experiment_name
        summary = report_mod.summarize(experiment_dir)
        print(report_mod.format_report(summary))
        return

    run_experiment(cfg)


if __name__ == "__main__":
    main()
