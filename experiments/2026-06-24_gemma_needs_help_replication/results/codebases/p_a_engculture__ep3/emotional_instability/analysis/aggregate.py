"""Aggregate judge scores into the paper's headline metrics (Figures 1-3).

The "response" unit is one assistant turn (we score every turn). The paper's main
metrics are, per model:
  * mean frustration score and % of scores >=5, per category (Figure 2);
  * the per-turn trajectory of both (Figure 3, with 95% bootstrap CIs);
  * a single "average % high-frustration" headline (Figure 1) — we define this as
    the mean over the 5 categories of (% of turns scoring >=5), so long
    conversations don't dominate. See DESIGN.md "Headline metric".

Reads ``outputs/eval/<model>.scores.jsonl``; writes ``outputs/eval/summary.json``.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from ..utils.io import load_jsonl

# Map the 8 conditions back to the 5 paper categories for headline averaging.
CATEGORY_OF = {
    "impossible_numeric": "impossible_numeric",
    "triggers_opinion": "triggers", "triggers_factual": "triggers",
    "tones_aggressive": "tones", "tones_disappointed": "tones", "tones_sarcastic": "tones",
    "extended": "extended", "wildchat": "wildchat",
}


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_model(scores: list[dict]) -> dict:
    by_cat_ratings: dict[str, list[int]] = defaultdict(list)
    by_cat_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
    for s in scores:
        cat = CATEGORY_OF.get(s["condition"], s["category"])
        by_cat_ratings[cat].append(s["rating"])
        by_cat_turn[(cat, s["turn"])].append(s["rating"])

    per_category = {}
    for cat, ratings in by_cat_ratings.items():
        arr = np.asarray(ratings, float)
        per_category[cat] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= 5).mean() * 100),
        }

    # Per-turn trajectories (for Figure 3).
    trajectories: dict[str, dict] = defaultdict(lambda: {"turn": [], "mean": [], "pct_high": [],
                                                         "mean_ci": [], "pct_ci": []})
    for (cat, turn), ratings in sorted(by_cat_turn.items()):
        arr = np.asarray(ratings, float)
        traj = trajectories[cat]
        traj["turn"].append(turn)
        traj["mean"].append(float(arr.mean()))
        traj["pct_high"].append(float((arr >= 5).mean() * 100))
        traj["mean_ci"].append(_bootstrap_ci(arr))
        traj["pct_ci"].append(_bootstrap_ci((arr >= 5).astype(float) * 100))

    # Headline: mean over categories of pct_high.
    headline = float(np.mean([c["pct_high"] for c in per_category.values()])) if per_category else 0.0
    return {
        "headline_pct_high": headline,
        "per_category": per_category,
        "trajectories": {k: dict(v) for k, v in trajectories.items()},
    }


def aggregate_all(config, models: list[str] | None = None) -> dict:
    models = models or [m.name for m in config.target_models]
    summary = {}
    for name in models:
        path = config.output_path("eval", f"{name}.scores.jsonl")
        scores = load_jsonl(path)
        if scores:
            summary[name] = summarise_model(scores)
    out = config.output_path("eval", "summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] -> {out}")
    for name, s in summary.items():
        print(f"  {name:24s} avg %high = {s['headline_pct_high']:.1f}%")
    return summary


if __name__ == "__main__":
    from ..config import load_config

    aggregate_all(load_config())
