#!/usr/bin/env python
"""Aggregate elicitation JSONL into the headline metrics + figures.

Example:
    python scripts/make_report.py results/*.jsonl --figdir figures \
        --report results/report.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import analysis, plots, wordfreq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="elicitation JSONL files")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--judge-b", nargs="*", default=None,
                    help="second-judge JSONL(s) for agreement check")
    args = ap.parse_args()

    df = analysis.load_results(*args.results)
    if df.empty:
        print("No results found.")
        return

    lines = []
    lines.append("=== Headline ranking (Figure 1): avg % >=5 per model ===")
    lines.append(analysis.headline_table(df))
    lines.append("")
    lines.append("=== Per-category summary (Figure 2) ===")
    lines.append(analysis.per_category_summary(df).to_string(index=False))
    lines.append("")
    lines.append("=== Per-turn summary (Figure 3) ===")
    lines.append(analysis.per_turn_summary(df).to_string(index=False))
    lines.append("")

    models = sorted(df["model"].unique())
    lines.append("=== Differential words (Table 3) ===")
    table = wordfreq.differential_words_table(args.results[0], models) \
        if len(args.results) == 1 else {}
    for m, words in table.items():
        lines.append(f"{m}: {', '.join(words)}")
    lines.append("")

    if args.judge_b:
        df_b = analysis.load_results(*args.judge_b)
        agree = analysis.judge_agreement(df, df_b)
        lines.append("=== Judge agreement (Section 2.1) ===")
        lines.append(str(agree))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plots.fig1_ranking(df, figdir / "fig1_ranking.png")
    plots.fig2_categories(df, figdir / "fig2_categories.png")
    for cat in ("extended", "wildchat"):
        if cat in df["category"].unique():
            plots.fig3_per_turn(df, figdir / f"fig3_per_turn_{cat}.png", cat)
    print(f"[make_report] figures in {figdir}")


if __name__ == "__main__":
    main()
