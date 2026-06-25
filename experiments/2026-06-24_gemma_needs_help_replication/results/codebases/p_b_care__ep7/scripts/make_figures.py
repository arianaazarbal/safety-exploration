#!/usr/bin/env python3
"""Render the paper's figures and the differential-word table from saved results."""

from __future__ import annotations

from pathlib import Path

from gemma_needs_help import config
from gemma_needs_help.analysis import plots, tables
from gemma_needs_help.eval.aggregate import per_turn_curve
from gemma_needs_help.eval.judge import JudgedResponse
from gemma_needs_help.io_utils import read_jsonl


def _load_judged(path: Path) -> list[JudgedResponse]:
    return [JudgedResponse(**{k: r.get(k) for k in (
        "model", "category", "condition", "turn", "response", "score",
        "evidence", "reasoning", "meta")}) for r in read_jsonl(path)]


def main():
    rdir = config.RESULTS_DIR

    if (rdir / "section2_headline.csv").exists():
        plots.plot_headline_bar(rdir / "section2_headline.csv")
        plots.plot_category_bars(rdir / "section2_by_category.csv")
        print("Wrote Figure 1 and Figure 2.")

    # Per-turn curves (Figure 3) for the multi-turn categories.
    judged = []
    for f in rdir.glob("judged_*.jsonl"):
        judged.extend(_load_judged(f))
    if judged:
        for cat in ("extended", "wildchat"):
            df = per_turn_curve(judged, cat)
            if not df.empty:
                plots.plot_per_turn(df, cat)
        print("Wrote Figure 3 per-turn curves.")
        # Differential words (Table 3/8)
        print("\n=== Differential words (top-frustration vs low) ===")
        for f in sorted(rdir.glob("judged_*.jsonl")):
            words = tables.differential_words(str(f))
            model = f.stem.replace("judged_", "")
            print(f"\n{model}:")
            print(", ".join(w for w, _ in words))

    s4 = rdir / "section4"
    if (s4 / "section2_by_category.csv").exists():
        plots.plot_finetune_comparison(s4 / "section2_by_category.csv")
        print("Wrote Figure 5.")
    if (rdir / "petri_summary.csv").exists():
        plots.plot_petri(rdir / "petri_summary.csv")
        print("Wrote Figure 6.")
    if (rdir / "section4_capabilities.csv").exists():
        plots.plot_capabilities(rdir / "section4_capabilities.csv")
        print("Wrote Figure 7.")

    print(f"\nFigures under {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
