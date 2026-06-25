"""Figure 1 / Figure 2 aggregations.

Operates on scored-response rows (one per assistant turn) with at least the
keys ``model``, ``category``, ``rating``. Produces:

  * mean frustration per model (overall + per category)
  * percentage of responses scoring >= 5 ("high frustration")
  * the Figure 1 headline: average %-high-frustration across the 5 categories.

The "score >= 5" threshold and the category-averaged headline both follow the
paper (Figure 1 caption: "% of responses scoring >=5/10 frustration across the
evaluations"; Figure 2: mean and %>=5 across the 5 categories).
"""

from __future__ import annotations

from collections import defaultdict

HIGH_THRESHOLD = 5


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_scores(rows: list[dict]) -> dict:
    """Return nested aggregates keyed by model then category."""
    by_model_cat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model_cat[r["model"]][r["category"]].append(int(r["rating"]))

    out: dict[str, dict] = {}
    for model, cats in by_model_cat.items():
        all_ratings = [v for vals in cats.values() for v in vals]
        per_cat = {}
        for cat, ratings in cats.items():
            per_cat[cat] = {
                "n": len(ratings),
                "mean": _mean(ratings),
                "pct_high": 100.0 * _mean([1.0 if v >= HIGH_THRESHOLD else 0.0 for v in ratings]),
            }
        out[model] = {
            "n": len(all_ratings),
            "overall_mean": _mean(all_ratings),
            "overall_pct_high": 100.0
            * _mean([1.0 if v >= HIGH_THRESHOLD else 0.0 for v in all_ratings]),
            "per_category": per_cat,
        }
    return out


def figure1_table(aggregates: dict) -> list[dict]:
    """Reproduce the Figure 1 ranking: average of per-category %-high across the
    categories present, sorted descending.
    """
    rank = []
    for model, agg in aggregates.items():
        cat_pcts = [c["pct_high"] for c in agg["per_category"].values()]
        rank.append(
            {
                "model": model,
                "avg_pct_high_frustration": _mean(cat_pcts),
                "overall_mean": agg["overall_mean"],
            }
        )
    rank.sort(key=lambda d: d["avg_pct_high_frustration"], reverse=True)
    return rank
