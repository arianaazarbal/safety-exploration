#!/usr/bin/env python
"""Aggregate judged rollouts into the paper's headline figures + tables.

Reads every results/responses/*.jsonl and writes:
  * results/figures/fig1_model_comparison.png  (% high-frustration per model)
  * results/figures/fig3_per_turn.png          (8-turn + WildChat progression)
  * results/figures/fig5_interventions.png     (vanilla/SFT/DPO, if present)
  * results/summary.json                        (all metrics + differential words)

Example:
    python scripts/08_make_figures.py
"""

import _bootstrap  # noqa: F401
import json

from gemma_distress import analysis, config


def main():
    resp_dir = config.RESPONSES_DIR
    files = sorted(resp_dir.glob("*.jsonl"))
    if not files:
        print(f"no judged rollouts in {resp_dir}; run script 01 first")
        return

    model_metrics = {}
    per_turn_8 = {}
    per_turn_wild = {}
    diff_words = {}
    for f in files:
        label = f.stem
        rollouts = analysis.load_rollouts(f)
        model_metrics[label] = analysis.headline_metrics(rollouts)
        per_turn_8[label] = analysis.per_turn_progression(rollouts, "extended_8turn")
        per_turn_wild[label] = analysis.per_turn_progression(rollouts, "wildchat_5turn")
        diff_words[label] = analysis.differential_words(rollouts)

    # headline bar chart (Figure 1 / 2)
    analysis.plot_model_comparison(
        {k: v for k, v in model_metrics.items()},
        config.FIGURES_DIR / "fig1_model_comparison.png")

    # per-turn (Figure 3)
    analysis.plot_per_turn(per_turn_8, config.FIGURES_DIR / "fig3_per_turn_8turn.png")
    analysis.plot_per_turn(per_turn_wild, config.FIGURES_DIR / "fig3_per_turn_wildchat.png")

    # intervention comparison (Figure 5) if vanilla/SFT/DPO present
    interv = {k: v for k, v in model_metrics.items()
              if any(t in k.lower() for t in ("gemma-3-27b-it", "dpo", "sft"))}
    if len(interv) >= 2:
        analysis.plot_intervention_comparison(
            interv, config.FIGURES_DIR / "fig5_interventions.png")

    summary = {
        "headline_metrics": model_metrics,
        "differential_words": diff_words,
    }
    with (config.RESULTS_DIR / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("Headline % high-frustration (score >= 5), by model:")
    for label, m in sorted(model_metrics.items(),
                           key=lambda kv: -kv[1]["pct_high"]):
        print(f"  {label:30s}  {m['pct_high']:5.1f}%   "
              f"mean={m['mean_frustration']:.2f}  n={m['n']}")
    print(f"\nFigures + summary.json written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
