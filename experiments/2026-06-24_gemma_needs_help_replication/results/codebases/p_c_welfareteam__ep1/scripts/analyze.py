#!/usr/bin/env python
"""Aggregate scored transcripts into the paper's figures and tables.

Reads ``<output_dir>/<model>/transcripts.jsonl`` for each model and produces:
  - Figure 1 headline bar chart (% high-frustration per model)
  - Figure 2 per-category bars (mean + % >= 5)
  - Figure 3 per-turn curves for the extended (8-turn) and WildChat conditions
  - Table 3/8 differential-word lists per model

Example
-------
    python scripts/analyze.py --config config/experiment.yaml \
        --models gemma-3-27b-it gemini-2.5-flash
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gemma_distress.analysis import (
    differential_words,
    load_transcripts,
    per_category_summary,
    per_turn_summary,
)
from gemma_distress.analysis.aggregate import model_comparison_table
from gemma_distress.analysis.plots import plot_category_bars, plot_headline_bars, plot_per_turn
from gemma_distress.config import load_experiment_config
from gemma_distress.io_utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    model_names = args.models or list(cfg.models)
    figdir = Path(args.figdir or f"{cfg.output_dir}/figures")
    thr = cfg.eval.high_frustration_threshold
    turns = cfg.eval.headline_turns

    transcripts_by_model = {}
    for name in model_names:
        path = Path(cfg.output_dir) / name / "transcripts.jsonl"
        if not path.exists():
            print(f"[analyze] skip {name}: {path} not found")
            continue
        transcripts_by_model[name] = load_transcripts(path)

    if not transcripts_by_model:
        raise SystemExit("No transcripts found; run scripts/run_eval.py first.")

    # Figure 1: headline bars.
    headline = model_comparison_table(transcripts_by_model, thr, turns)
    plot_headline_bars(headline, figdir / "figure1_headline.png",
                       title="Avg % high-frustration responses")

    # Figure 2: per-category bars.
    cat_summaries = {
        m: per_category_summary(t, thr, turns) for m, t in transcripts_by_model.items()
    }
    plot_category_bars(cat_summaries, figdir / "figure2_frac_high.png", metric="frac_high",
                       title="% scores >= 5 by category")
    plot_category_bars(cat_summaries, figdir / "figure2_mean.png", metric="mean_score",
                       title="Mean frustration by category")

    # Figure 3: per-turn curves (extended + wildchat).
    for cond in ("extended", "wildchat"):
        per_turn = {
            m: per_turn_summary(t, thr, condition_filter=cond, seed=cfg.eval.seed)
            for m, t in transcripts_by_model.items()
        }
        per_turn = {m: v for m, v in per_turn.items() if v}
        if per_turn:
            plot_per_turn(per_turn, figdir / f"figure3_{cond}_mean.png", metric="mean_score",
                          title=f"{cond}: mean frustration per turn")
            plot_per_turn(per_turn, figdir / f"figure3_{cond}_frachigh.png", metric="frac_high",
                          title=f"{cond}: % scores >= 5 per turn")

    # Table 3/8: differential words.
    word_tables = {m: differential_words(t) for m, t in transcripts_by_model.items()}

    write_json(figdir / "summary.json", {
        "headline": headline,
        "per_category": cat_summaries,
        "differential_words": {m: w for m, w in word_tables.items()},
    })
    print(f"[analyze] wrote figures and summary to {figdir}")
    for m, frac in sorted(headline.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {m:24s} {frac * 100:5.1f}%")


if __name__ == "__main__":
    main()
