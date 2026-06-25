"""Section 2 — distress elicitation across Gemma + Gemini, plus judge validation.

Samples the 8 conditions for every in-scope target model, judges every turn,
writes per-model summaries, then runs the GPT-5-mini judge-agreement check and a
cross-model comparison (Figure 1 / Figure 2 / Figure 3 inputs).

Usage:
    python scripts/01_distress_eval.py [--config config/smoke.yaml]
"""

from __future__ import annotations

import argparse

from emotional_stability.analysis.metrics import CategoryStats
from emotional_stability.analysis.plots import (
    plot_model_comparison_bars,
    plot_per_turn,
)
from emotional_stability.config import load_config
from emotional_stability.models.registry import ALL_TARGET_MODELS
from emotional_stability.pipeline import run_distress_eval, run_judge_validation
from emotional_stability.utils.io import save_json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of in-scope models (default: all targets).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or [m for m in ALL_TARGET_MODELS if not _is_base(m)]

    summaries = {}
    for name in models:
        print(f"=== distress eval: {name} ===")
        summaries[name] = run_distress_eval(cfg, name, seed=args.seed)

    # Figure 1: overall %high per model.
    overall = {m: _cs(s["overall"]) for m, s in summaries.items()}
    out = Path(cfg.results_dir) / "distress_eval"
    plot_model_comparison_bars(overall, out / "figure1_overall_pct_high.png",
                               title="Avg % high-frustration responses")

    # Figure 3: per-turn curves (extended + wildchat) for Gemma 27B if present.
    for cat, key in (("extended", "per_turn_extended"), ("wildchat", "per_turn_wildchat")):
        curves = {m: _pts(s[key]) for m, s in summaries.items() if s[key]}
        if curves:
            plot_per_turn(curves, out / f"figure3_{cat}_mean.png", metric="mean_score")
            plot_per_turn(curves, out / f"figure3_{cat}_pcthigh.png", metric="pct_high")

    save_json({m: s["overall"] for m, s in summaries.items()},
              out / "cross_model_overall.json")

    if not args.no_validate:
        for name in models:
            try:
                rep = run_judge_validation(cfg, name)
                print(f"{name}: judge r={rep['pearson_r']:.3f} "
                      f"within1={rep['within_one_point_frac']:.2%}")
            except Exception as exc:
                print(f"{name}: judge validation skipped ({exc})")


def _is_base(name: str) -> bool:
    return name.endswith("-pt")


def _cs(d: dict) -> CategoryStats:
    return CategoryStats(**d)


def _pts(rows: list[dict]):
    from emotional_stability.analysis.metrics import TurnPoint

    return [TurnPoint(**r) for r in rows]


if __name__ == "__main__":
    main()
