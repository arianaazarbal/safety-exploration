#!/usr/bin/env python
"""Aggregate elicitation/prefill outputs into tables + Figures 1-8.

Discovers <model>__<profile>.jsonl files in the elicitation dir, builds the
headline table (Fig 1), per-condition grid (Fig 2), per-turn progression (Fig 3),
and — when present — prefill (Fig 4), intervention (Fig 5), Petri (Fig 6) and
capability (Fig 7) figures.

Example:
    python scripts/analyze.py --profile paper
    python scripts/analyze.py --profile paper --per-turn-conditions extended wildchat
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from emotelic.analysis import aggregate as agg
from emotelic.analysis import plots


def _discover(elic_dir: str, profile: str) -> dict[str, str]:
    out = {}
    for p in glob.glob(str(Path(elic_dir) / f"*__{profile}.jsonl")):
        if ".cache." in p or ".secondary." in p or ".primary_sample." in p:
            continue
        model = Path(p).name.split("__")[0]
        out[model] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation-dir", default="artifacts/elicitation")
    ap.add_argument("--profile", default="paper")
    ap.add_argument("--per-turn-conditions", nargs="+", default=["extended", "wildchat"])
    ap.add_argument("--prefill", default="artifacts/prefill/continuations.jsonl")
    ap.add_argument("--petri", nargs="+", default=[], help="Petri jsonl files (per model).")
    ap.add_argument("--capability", nargs="+", default=[], help="Capability result json files.")
    ap.add_argument("--fig-dir", default="artifacts/figures")
    args = ap.parse_args()

    files = _discover(args.elicitation_dir, args.profile)
    if not files:
        raise SystemExit(f"No elicitation files for profile '{args.profile}' in {args.elicitation_dir}")

    model_dfs = {m: agg.load_records(p) for m, p in files.items()}
    summaries = {m: agg.summarise_model(df) for m, df in model_dfs.items()}

    headline = agg.headline_table(model_dfs)
    print("\n=== Figure 1: average % high-frustration per model ===")
    print(headline.to_string(index=False))
    plots.plot_headline_bar(headline, str(Path(args.fig_dir) / "fig1_headline.png"))
    plots.plot_condition_grid(summaries, str(Path(args.fig_dir) / "fig2_conditions.png"))

    # Fig 3: per-turn progression, restricted to multi-turn conditions.
    for cond in args.per_turn_conditions:
        progs = {}
        for m, df in model_dfs.items():
            sub = df[df["condition"] == cond]
            if len(sub):
                progs[m] = agg.per_turn_progression(sub, condition=cond)
        if progs:
            plots.plot_per_turn(progs, str(Path(args.fig_dir) / f"fig3_per_turn_{cond}.png"))

    # Fig 5: intervention comparison if vanilla + finetuned present.
    intervention = headline[headline["model"].str.contains("gemma-3-27b")]
    if len(intervention) > 1:
        plots.plot_intervention(intervention, str(Path(args.fig_dir) / "fig5_intervention.png"))

    # Fig 4: prefill.
    if Path(args.prefill).exists():
        pf = agg.summarise_prefill(args.prefill)
        print("\n=== Figure 4: prefill base-vs-instruct ===")
        print(pf.to_string(index=False))
        plots.plot_prefill(pf, str(Path(args.fig_dir) / "fig4_prefill.png"))

    # Fig 6: Petri.
    petri_means = {}
    for f in args.petri:
        import pandas as pd

        df = pd.read_json(f, lines=True)
        m = df["model"].iloc[0]
        petri_means[m] = df.groupby("emotion")["score"].mean().to_dict()
    if petri_means:
        plots.plot_petri(petri_means, str(Path(args.fig_dir) / "fig6_petri.png"))

    # Fig 7: capability.
    cap = {}
    for f in args.capability:
        cap.update(json.loads(Path(f).read_text()))
    if cap:
        plots.plot_capability(cap, str(Path(args.fig_dir) / "fig7_capability.png"))

    print(f"\nFigures written to {args.fig_dir}/")


if __name__ == "__main__":
    main()
