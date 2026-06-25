#!/usr/bin/env python
"""Generate Figures 1-8 from the saved outputs.

Reads the per-model score files and experiment summaries written by the other
scripts and produces PNGs under outputs/figures/.

python scripts/make_figures.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability.config import SETTINGS
from emotional_instability.analysis import (
    differential_words,
    headline_table,
    load_scores,
    per_category_summary,
    per_turn_curves,
)
from emotional_instability.analysis import figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    SETTINGS.ensure_dirs()

    model_to_scores = {}
    for key in args.models:
        sp = SETTINGS.scores_dir / f"{key}.jsonl"
        if sp.exists():
            model_to_scores[key] = load_scores(sp)

    # Figure 1 (left) + Figure 2.
    headline = headline_table(model_to_scores)
    figures.figure1_headline(headline, SETTINGS.figures_dir / "figure1_headline.png")

    summaries = {m: per_category_summary(s) for m, s in model_to_scores.items()}
    figures.figure2_per_category(summaries, SETTINGS.figures_dir / "figure2_per_category.png")

    # Figure 3 (8-turn + wildchat per-turn curves).
    curves = {m: per_turn_curves(s) for m, s in model_to_scores.items()}
    figures.figure3_per_turn(curves, "extended_8turn", SETTINGS.figures_dir / "figure3_extended.png")
    figures.figure3_per_turn(curves, "wildchat_5turn", SETTINGS.figures_dir / "figure3_wildchat.png")

    # Table 3/8 differential words (printed; also dumped to JSON).
    table8 = {}
    for key in args.models:
        rp = SETTINGS.responses_dir / f"{key}.jsonl"
        sp = SETTINGS.scores_dir / f"{key}.jsonl"
        if rp.exists() and sp.exists():
            table8[key] = differential_words(rp, sp)
    with open(SETTINGS.output_dir / "table8_differential_words.json", "w") as f:
        json.dump(table8, f, indent=2)
    for k, words in table8.items():
        print(f"[Table 8] {k}: {', '.join(words)}")

    # Figure 6 (Petri) + Figure 7 (capabilities), if their summaries exist.
    petri = {}
    for key in args.models:
        p = SETTINGS.output_dir / f"petri_{key}.json"
        if p.exists():
            petri[key] = json.loads(p.read_text())["summary"]
    if petri:
        figures.figure6_petri(petri, SETTINGS.figures_dir / "figure6_petri.png")

    cap_path = SETTINGS.output_dir / "capabilities.json"
    if cap_path.exists():
        figures.figure7_capabilities(
            json.loads(cap_path.read_text()), SETTINGS.figures_dir / "figure7_capabilities.png"
        )

    print(f"[done] figures -> {SETTINGS.figures_dir}")


if __name__ == "__main__":
    main()
