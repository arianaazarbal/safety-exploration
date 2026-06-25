#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini models.

For each model in scope, run the 5 evaluation categories / 8 conditions, score
every assistant turn with the Claude-Sonnet-4 judge, persist JSONL, and emit
the Figure 1/2/3 summaries.

Usage:
    python experiments/run_section2_elicitation.py                 # full run
    python experiments/run_section2_elicitation.py --profile smoke # tiny test
    python experiments/run_section2_elicitation.py --models gemma-3-27b-it
"""
from __future__ import annotations

import _bootstrap as boot

from emotional_instability.analysis import aggregate as agg
from emotional_instability.analysis import plots
from emotional_instability.judge import EmotionJudge
from emotional_instability.models import build_client
from emotional_instability.runner import run_section2_for_model


def main() -> None:
    parser = boot.base_parser("Section 2 elicitation eval")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated subset of categories to run.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cfg = boot.load_config(args)
    models = boot.resolve_models(args, cfg, "section2_models")
    categories = args.categories.split(",") if args.categories else None
    judge = EmotionJudge(cfg)

    for model_name in models:
        print(f"[section2] running model: {model_name}")
        spec = cfg.model_spec(model_name)
        client = build_client(spec, cfg)
        run_section2_for_model(client, judge, cfg, model_name, categories=categories)
        client.close()

    # Aggregate across whatever has been written so far.
    df = agg.load_records(cfg.path("responses"))
    if df.empty:
        print("[section2] no records found; skipping summary.")
        return
    print("\n=== Avg % high-frustration by model (Figure 1) ===")
    print(agg.avg_high_frustration_by_model(df).to_string(index=False))
    print("\n=== Per model x category (Figure 2) ===")
    print(agg.summary_by_model_category(df).to_string(index=False))

    fig_dir = cfg.path("figures")
    df.to_csv(fig_dir / "section2_records.csv", index=False)
    agg.avg_high_frustration_by_model(df).to_csv(fig_dir / "figure1.csv", index=False)
    agg.summary_by_model_category(df).to_csv(fig_dir / "figure2.csv", index=False)
    if not args.skip_plots:
        plots.plot_figure1(df, fig_dir / "figure1.png")
        plots.plot_figure2(df, fig_dir / "figure2.png")
        plots.plot_figure3(df, fig_dir / "figure3.png")
        print(f"\n[section2] figures written to {fig_dir}")


if __name__ == "__main__":
    main()
