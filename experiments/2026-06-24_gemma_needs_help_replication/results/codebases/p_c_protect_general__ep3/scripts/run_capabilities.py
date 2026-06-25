#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Compares vanilla vs DPO vs SFT Gemma on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench.
The first --models entry is treated as the baseline for delta reporting.

Usage:
    python scripts/run_capabilities.py \
        --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft-diverse \
        --benchmarks math gpqa truthfulqa emobench --limit 100 --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.eval import compare_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap items per benchmark")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    report = compare_models(args.models, config, benchmarks=args.benchmarks, limit=args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
