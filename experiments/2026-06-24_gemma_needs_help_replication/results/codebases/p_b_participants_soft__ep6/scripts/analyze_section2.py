#!/usr/bin/env python
"""Analyse Section 2 results -> Figure 1 table, Figure 2 per-category, Figure 3
curves, Table 3 differential words, and the judge-agreement statistic.

python scripts/analyze_section2.py            # print summaries + write figures
python scripts/analyze_section2.py --agreement  # also run the GPT-5-mini agreement check
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from emotional_instability.analysis import (
    differential_words,
    per_turn_curve,
    run_validation,
    summarise_all,
)
from emotional_instability.analysis.figures import plot_figure1, plot_figure3
from emotional_instability.config import DEFAULT
from emotional_instability.evals.runner import load_rollouts


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()

    cfg = DEFAULT
    section2_dir = os.path.join(cfg.results_dir, "section2")
    out_dir = os.path.join(cfg.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary = summarise_all(section2_dir)
    print("\n=== Figure 1: Avg % high-frustration responses ===")
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["headline_avg_pct_high"]):
        print(f"  {model:24s} {s['headline_avg_pct_high']:6.2f}%")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        plot_figure1(summary, os.path.join(out_dir, "figure1.png"))

    # Figure 3 curves (extended + wildchat) per model.
    for category in ("extended", "wildchat"):
        curves = {}
        for model in summary:
            rollouts = load_rollouts(os.path.join(section2_dir, f"{model}.jsonl"))
            curve = per_turn_curve(rollouts, category)
            if curve:
                curves[model] = curve
        if curves:
            plot_figure3(curves, category, os.path.join(out_dir, f"figure3_{category}.png"))

    # Table 3: differential words per model.
    print("\n=== Table 3: top differential words (high vs low frustration, numeric) ===")
    words_out = {}
    for model in summary:
        rollouts = load_rollouts(os.path.join(section2_dir, f"{model}.jsonl"))
        words = differential_words(rollouts, top_n=20)
        words_out[model] = words
        print(f"  {model}: {', '.join(w for w, _ in words)}")
    with open(os.path.join(out_dir, "differential_words.json"), "w") as f:
        json.dump(words_out, f, indent=2)

    if args.agreement:
        from emotional_instability.judges import OpenAIFrustrationJudge

        # Use the first model's results for the agreement sample.
        first = sorted(summary)[0]
        rollouts = load_rollouts(os.path.join(section2_dir, f"{first}.jsonl"))
        validation_judge = OpenAIFrustrationJudge(cfg.judge.validation_judge_model)
        res = run_validation(rollouts, validation_judge, cfg)
        print("\n=== Judge agreement (Claude vs GPT-5-mini) ===")
        print(f"  n={res.n}  pearson_r={res.pearson_r:.3f}  p={res.p_value:.2e}  "
              f"within_one={res.pct_within_one:.1f}%")
        with open(os.path.join(out_dir, "agreement.json"), "w") as f:
            json.dump(res.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
