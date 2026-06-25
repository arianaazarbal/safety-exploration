"""Generate the paper's core figures from saved results.

    python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from distress_eval import aggregate, plots
from distress_eval.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--convention", default="final", choices=["final", "max"])
    args = ap.parse_args()

    config = load_config(args.config)
    fig_dir = config.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate.load_responses(config.output_dir / "responses")
    if not df.empty:
        summary = aggregate.per_model_summary(df, convention=args.convention)
        plots.plot_model_summary(summary, fig_dir / "figure2_model_summary.png")

        progression = aggregate.per_turn_progression(df, categories=["extended", "wildchat"])
        if not progression.empty:
            plots.plot_per_turn(progression, fig_dir / "figure3_extended.png", "extended")
            plots.plot_per_turn(progression, fig_dir / "figure3_wildchat.png", "wildchat")

        # Figure 5: intervention comparison (vanilla / SFT / DPO Gemma)
        keys = [k for k in ["gemma-3-27b-it", "gemma-3-27b-sft", "gemma-3-27b-dpo"]
                if k in set(summary["model_key"])]
        if len(keys) >= 2:
            plots.plot_intervention_comparison(summary, fig_dir / "figure5_intervention.png", keys)

        print(summary.to_string(index=False))

    petri_path = config.output_dir / "petri" / "transcripts.jsonl"
    if petri_path.exists():
        rows = [json.loads(l) for l in petri_path.read_text().splitlines() if l.strip()]
        pdf = pd.DataFrame(rows)
        if not pdf.empty:
            plots.plot_petri(pdf, fig_dir / "figure6_petri.png")

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
