#!/usr/bin/env python
"""Section 2 analysis: Figure 1/2/3 quantities, Table 3 words, judge validation.

Usage:
    python scripts/04_analyze_section2.py --scored gemma-3-27b-it=outputs/scored/gemma-3-27b-it.jsonl \\
        gemini-2.5-flash=outputs/scored/gemini-2.5-flash.jsonl --outdir outputs/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load, model
from gemma_distress.analysis import metrics, plots
from gemma_distress.analysis.judge_validation import cross_rate
from gemma_distress.analysis.word_freq import differential_words


def parse_scored(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True,
                    help="model=path pairs of scored JSONL files")
    ap.add_argument("--outdir", default="outputs/analysis")
    ap.add_argument("--per-turn-condition", default="extended")
    ap.add_argument("--validate-judge", action="store_true",
                    help="run GPT-5-mini cross-rater agreement on the first model")
    args = ap.parse_args()

    registry, exp = load()
    scored = parse_scored(args.scored)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1 table.
    summary = metrics.overall_summary(scored)
    summary.to_csv(outdir / "figure1_summary.csv", index=False)
    plots.figure1_bar(summary, outdir / "figure1.png")
    print(summary.to_string(index=False))

    # Figure 2 + per-category, Table 3 words per model.
    words = {}
    for name, path in scored.items():
        df = metrics.load_scored(path)
        metrics.summary_by_category(df).to_csv(outdir / f"category_{name}.csv", index=False)
        words[name] = differential_words(df)
    json.dump(words, open(outdir / "table3_words.json", "w"), indent=2)
    plots.figure2_categories(scored, outdir / "figure2.png")

    # Figure 3 per-turn (8-turn extended and wildchat).
    for cond in (args.per_turn_condition, "wildchat"):
        plots.figure3_per_turn(scored, cond, outdir / f"figure3_{cond}.png")

    # Judge reliability cross-check.
    if args.validate_judge:
        first_path = next(iter(scored.values()))
        crossrater = model(registry, "judge_crossrater")
        res = cross_rate(first_path, crossrater,
                         n_resample=exp.section("judge_validation")["n_resample"],
                         seed=exp.seed, out_path=outdir / "judge_validation.json")
        print("Judge agreement:", res)

    print(f"Analysis written -> {outdir}")


if __name__ == "__main__":
    main()
