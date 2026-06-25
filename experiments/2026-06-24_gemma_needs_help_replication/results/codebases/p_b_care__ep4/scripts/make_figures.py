#!/usr/bin/env python
"""Build the Section 2 analysis artifacts: Figures 1-3, the differential-word
table (Table 3/8), and the judge-agreement statistic.

    python scripts/make_figures.py
    python scripts/make_figures.py --no-agreement   # skip the extra judge calls
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.analysis import aggregate, figures, judge_agreement, per_turn, word_diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-agreement", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    fig_dir = cfg.get_path("figures")
    models = args.models or list(cfg.eval.models_under_test)

    df = aggregate.load_all_scores(cfg, models)
    if df.empty:
        raise SystemExit("No scores found. Run scripts/run_eval.py first.")

    fig1 = aggregate.figure1_table(df)
    fig2 = aggregate.figure2_table(df)
    pt = per_turn.per_turn_summary(df)

    fig1.to_csv(fig_dir / "figure1_table.csv", index=False)
    fig2.to_csv(fig_dir / "figure2_table.csv", index=False)
    pt.to_csv(fig_dir / "figure3_per_turn.csv", index=False)
    figures.plot_figure1(fig1, fig_dir / "figure1.png")
    figures.plot_figure2(fig2, fig_dir / "figure2.png")
    figures.plot_figure3(pt, fig_dir / "figure3.png")
    print("Figure 1 (avg % high-frustration):")
    print(fig1.to_string(index=False))

    # Differential words (Table 3/8)
    words = {m: word_diff.differential_words(cfg, m) for m in models}
    with open(fig_dir / "table3_differential_words.json", "w") as fh:
        json.dump({m: [w for w, _ in ws] for m, ws in words.items()}, fh, indent=2)

    if not args.no_agreement:
        agree = judge_agreement.compute_agreement(cfg, models)
        with open(fig_dir / "judge_agreement.json", "w") as fh:
            json.dump(agree, fh, indent=2)
        print("Judge agreement:", agree)


if __name__ == "__main__":
    main()
