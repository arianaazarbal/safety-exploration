#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation (Appendix G).

Usage:
    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash \
        --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.eval import run_petri_emotion_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    all_results = {}
    for model in args.models:
        print(f"=== Petri: {model} ===")
        all_results[model] = run_petri_emotion_eval(model, config)
        print(json.dumps(all_results[model], indent=2))

    out = Path(config.output_dir) / "petri" / "all_models_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
