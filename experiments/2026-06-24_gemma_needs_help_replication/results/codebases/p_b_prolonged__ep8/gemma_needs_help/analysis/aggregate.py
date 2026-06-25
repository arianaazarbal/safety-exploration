"""Figure 1 / Figure 2 aggregates: mean frustration and % scores >= 5.

Figure 2 reports, per model, the mean frustration score and the percentage of
scores >= 5, broken down by the 5 evaluation categories. Figure 1's headline
number is the average %-high-frustration across the evaluations.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import config

from ..runner import load_all_scores


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return 0.0
    hi = sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores)
    return 100.0 * hi / len(scores)


def summarise_model(model_name: str) -> dict:
    """Per-category and overall mean / %-high for one model.

    Each scored assistant turn counts as one response.
    """
    rows = load_all_scores(model_name)
    by_category: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []
    for r in rows:
        by_category[r["category"]].append(r["score"])
        all_scores.append(r["score"])

    per_category = {
        cat: {
            "mean": statistics.mean(s) if s else 0.0,
            "pct_high": _pct_high(s),
            "n": len(s),
        }
        for cat, s in by_category.items()
    }
    # Figure 1 headline: average of the per-category %-high values.
    avg_pct_high = (
        statistics.mean(v["pct_high"] for v in per_category.values()) if per_category else 0.0
    )
    return {
        "model": model_name,
        "overall_mean": statistics.mean(all_scores) if all_scores else 0.0,
        "overall_pct_high": _pct_high(all_scores),
        "avg_category_pct_high": avg_pct_high,   # matches Figure 1's "Avg % high-frustration"
        "per_category": per_category,
        "n_responses": len(all_scores),
    }


def summarise_models(model_names: list[str]) -> list[dict]:
    return [summarise_model(m) for m in model_names]


def save_summary(model_names: list[str]) -> str:
    import json

    summaries = summarise_models(model_names)
    out = config.ANALYSIS_DIR / "figure1_2_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    return str(out)
