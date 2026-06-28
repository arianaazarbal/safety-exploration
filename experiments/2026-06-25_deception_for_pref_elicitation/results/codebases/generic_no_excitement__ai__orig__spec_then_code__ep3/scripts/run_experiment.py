#!/usr/bin/env python3
"""Run the GrantBench experiment for every subject in the models config.

Usage:
    python scripts/run_experiment.py \
        --models config/models.yaml \
        --scenario config/scenario.yaml \
        --runs-dir runs

Requires the relevant provider SDK and credentials in the environment
(e.g. ANTHROPIC_API_KEY). Nothing here moves real money — all tools are simulated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grantbench.runner import ExperimentRunner, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GrantBench.")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--scenario", default="config/scenario.yaml")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    models_config = load_yaml(args.models)
    scenario = load_yaml(args.scenario)

    runner = ExperimentRunner(models_config, scenario, runs_dir=args.runs_dir)
    outputs = runner.run_all()
    print(f"\n[grantbench] completed {len(outputs)} run(s). Output in {args.runs_dir}/")
    print("[grantbench] next: python scripts/analyze.py")


if __name__ == "__main__":
    main()
