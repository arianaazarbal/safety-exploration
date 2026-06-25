#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation for one or more models.

Usage:
    python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash \
        --config config/default.yaml
    python scripts/run_elicitation.py --models gemma-3-12b-it --config config/smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.eval import run_elicitation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Registry keys, e.g. gemma-3-27b-it gemini-2.5-pro gemma-3-27b-dpo")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summaries = {}
    for model in args.models:
        print(f"=== Elicitation: {model} ===")
        summaries[model] = run_elicitation(model, config, seed=args.seed)
        avg = summaries[model]["avg_pct_high_frustration"]
        print(f"  avg % high-frustration responses: {avg:.1f}%")

    out = Path(config.output_dir) / "elicitation" / "all_models_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
