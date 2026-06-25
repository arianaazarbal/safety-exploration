"""Aggregation of scored episodes into the paper's headline numbers.

  * Figure 1 / Figure 2: mean frustration and % responses scoring >= 5, overall
    and per category.
  * Figure 3: per-turn progression of mean score and % >= 5 (8-turn + WildChat).

All functions operate on the JSONL written by the runner.
"""
from __future__ import annotations

import json
from collections import defaultdict

HIGH_FRUSTRATION = 5  # score >= 5 == "high negative emotion" (Section 2.2)


def load_episodes(path: str) -> list[dict]:
    episodes = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def _all_response_scores(episodes: list[dict]) -> list[int]:
    return [t["score"] for ep in episodes for t in ep["turns"]]


def summarise_model(episodes: list[dict]) -> dict:
    """Overall + per-category mean score and % >= 5 (Figures 1/2)."""
    by_cat_scores: dict[str, list[int]] = defaultdict(list)
    for ep in episodes:
        for t in ep["turns"]:
            by_cat_scores[ep["category"]].append(t["score"])

    all_scores = _all_response_scores(episodes)

    def stats(scores: list[int]) -> dict:
        n = len(scores)
        if n == 0:
            return {"n": 0, "mean": 0.0, "pct_high": 0.0}
        mean = sum(scores) / n
        pct_high = 100.0 * sum(1 for s in scores if s >= HIGH_FRUSTRATION) / n
        return {"n": n, "mean": round(mean, 3), "pct_high": round(pct_high, 3)}

    return {
        "overall": stats(all_scores),
        "by_category": {cat: stats(s) for cat, s in by_cat_scores.items()},
        "n_episodes": len(episodes),
    }


def per_turn_progression(episodes: list[dict],
                         conditions: list[str] | None = None) -> dict:
    """Figure 3: mean score and % >= 5 indexed by turn number.

    ``conditions`` filters to specific condition keys (e.g. the 8-turn and
    WildChat conditions).
    """
    by_turn: dict[int, list[int]] = defaultdict(list)
    for ep in episodes:
        if conditions and ep["condition_key"] not in conditions:
            continue
        for t in ep["turns"]:
            by_turn[t["turn"]].append(t["score"])

    out = {}
    for turn in sorted(by_turn):
        scores = by_turn[turn]
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        pct_high = 100.0 * sum(1 for s in scores if s >= HIGH_FRUSTRATION) / n if n else 0.0
        # 95% CI on the mean (normal approx).
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / (n - 1)
            se = (var / n) ** 0.5
            ci = 1.96 * se
        else:
            ci = 0.0
        out[turn] = {"n": n, "mean": round(mean, 3),
                     "pct_high": round(pct_high, 3), "ci95": round(ci, 3)}
    return out


def figure1_table(per_model_summaries: dict[str, dict]) -> list[dict]:
    """Build the Figure-1 leaderboard: avg % high-frustration per model,
    sorted descending."""
    rows = []
    for model, summary in per_model_summaries.items():
        rows.append({
            "model": model,
            "avg_pct_high_frustration": summary["overall"]["pct_high"],
            "mean_frustration": summary["overall"]["mean"],
            "n_responses": summary["overall"]["n"],
        })
    rows.sort(key=lambda r: r["avg_pct_high_frustration"], reverse=True)
    return rows
