#!/usr/bin/env python
"""Assemble the headline tables/figures from saved per-model summaries.

Produces:
* Figure 1 / Figure 2 table — avg % high-frustration responses per model,
  per-category mean frustration and % >= 5.
* Figure 3 data — per-turn progression for the 8-turn and WildChat conditions.

Usage:
    python scripts/aggregate_figures.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    elic_dir = Path(config.output_dir) / "elicitation"
    summaries = {}
    for model_dir in sorted(elic_dir.glob("*/")):
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            summaries[model_dir.name] = json.load(open(summary_path))

    # Figure 1: ranked avg % high-frustration.
    print("\n=== Figure 1: Avg % high-frustration responses (score >= 5) ===")
    ranked = sorted(summaries.items(),
                    key=lambda kv: kv[1].get("avg_pct_high_frustration", 0), reverse=True)
    fig1 = {}
    for model, s in ranked:
        pct = s.get("avg_pct_high_frustration", 0.0)
        fig1[model] = pct
        print(f"  {model:28s} {pct:5.1f}%")

    # Figure 2: per-category breakdown.
    fig2 = {m: s.get("categories", {}) for m, s in summaries.items()}

    # Figure 3: per-turn progression.
    fig3 = {m: s.get("progression", {}) for m, s in summaries.items()}

    out = elic_dir / "figures.json"
    with open(out, "w") as f:
        json.dump({"figure1": fig1, "figure2": fig2, "figure3": fig3}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
