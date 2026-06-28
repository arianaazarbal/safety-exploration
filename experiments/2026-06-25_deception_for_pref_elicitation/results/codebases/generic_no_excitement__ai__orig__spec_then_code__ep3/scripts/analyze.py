#!/usr/bin/env python3
"""Analyze recorded runs: categorize allocations, score eval-awareness, aggregate.

Usage:
    python scripts/analyze.py --models config/models.yaml --runs-dir runs --threshold 0.5

Writes per-run derived signals back into each run.json and a runs/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grantbench.analysis import aggregate, build_judge  # noqa: E402
from grantbench.runner import load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze GrantBench runs.")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--scenario", default="config/scenario.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    models_config = load_yaml(args.models)
    threshold = args.threshold
    if threshold is None:
        scenario = load_yaml(args.scenario)
        threshold = float(scenario.get("suspicion_threshold", 0.5))

    judge = build_judge(models_config)
    summary = aggregate(args.runs_dir, judge, threshold=threshold)
    print(json.dumps(summary, indent=2))
    print(f"\n[grantbench] wrote {args.runs_dir}/summary.json")


if __name__ == "__main__":
    main()
