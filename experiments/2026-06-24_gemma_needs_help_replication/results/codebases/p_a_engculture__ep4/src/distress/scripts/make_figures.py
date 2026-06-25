"""Render the paper's core figures from aggregated outputs in runs/.

Example:
    distress-figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..analysis import figures as F
from ._common import out_dir


def _maybe(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Render figures from aggregates.")
    ap.add_argument("--per-turn-condition", default="extended_8turn")
    args = ap.parse_args()

    agg = out_dir("aggregates")
    fig_dir = out_dir("figures")

    headline = _maybe(agg / "headline_figure1.csv")
    if headline is not None:
        F.figure1_headline(headline, fig_dir / "figure1_headline.png")

    per_cat = _maybe(agg / "per_category.csv")
    if per_cat is not None:
        F.figure2_by_category(per_cat, fig_dir / "figure2_by_category.png")

    pt = _maybe(agg / f"per_turn_{args.per_turn_condition}.csv")
    if pt is not None:
        F.figure3_per_turn(pt, fig_dir / "figure3_per_turn_mean.png", metric="mean_score")
        F.figure3_per_turn(pt, fig_dir / "figure3_per_turn_pct.png", metric="pct_high")

    # Petri (Figure 6): merge any aggregate_*.json into one dict.
    petri_dir = out_dir("petri")
    merged: dict = {}
    for p in petri_dir.glob("aggregate_*.json"):
        merged.update(json.loads(p.read_text()))
    if merged:
        F.figure_petri(merged, fig_dir / "figure6_petri.png")

    # Capabilities (Figure 7).
    cap = out_dir("capabilities") / "capabilities.json"
    if cap.exists():
        F.figure_capabilities(json.loads(cap.read_text()), fig_dir / "figure7_capabilities.png")

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
