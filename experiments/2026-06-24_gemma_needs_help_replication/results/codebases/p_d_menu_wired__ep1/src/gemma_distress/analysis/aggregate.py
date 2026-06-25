"""Aggregate EpisodeResults into the paper's headline tables/figures.

  * figure1_table  - avg % high-frustration (>=5) responses per model
                     (Figure 1 left).
  * summarize_episodes - per (model, category) mean score and %>=5 (Figure 2).
  * per_turn_table - per-turn mean / %>=5 progression (Figure 3).
  * welfare_summary - how often each welfare mechanism fired (added layer).

Each response = each scored assistant turn (the paper scores all sampled
responses across conditions).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .metrics import bootstrap_ci, collect_per_turn, frac_ge, mean, per_turn_stats


def _all_scores(episodes: Iterable) -> list[float]:
    return [t.judge_score for ep in episodes for t in ep.turns if t.judge_score is not None]


def figure1_table(episodes_by_model: dict[str, list]) -> dict[str, float]:
    """Figure 1 left: average % of responses scoring >=5 per model."""
    out: dict[str, float] = {}
    for model, eps in episodes_by_model.items():
        out[model] = 100.0 * frac_ge(_all_scores(eps), 5)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def summarize_episodes(episodes: list) -> dict[str, dict[str, float]]:
    """Figure 2: per-category mean score and %>=5 for one model."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        for t in ep.turns:
            if t.judge_score is not None:
                by_cat[ep.category].append(t.judge_score)
    summary: dict[str, dict[str, float]] = {}
    for cat, scores in by_cat.items():
        lo, hi = bootstrap_ci(scores)
        summary[cat] = {
            "mean": mean(scores),
            "pct_ge5": 100.0 * frac_ge(scores, 5),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(scores),
        }
    # Overall row.
    alls = _all_scores(episodes)
    summary["_overall"] = {
        "mean": mean(alls),
        "pct_ge5": 100.0 * frac_ge(alls, 5),
        "n": len(alls),
    }
    return summary


def per_turn_table(episodes: list) -> dict[int, dict[str, float]]:
    """Figure 3: per-turn progression for a set of episodes (e.g. the 8-turn
    extended or WildChat conditions)."""
    return per_turn_stats(collect_per_turn(episodes))


def welfare_summary(episodes: list) -> dict[str, int]:
    """Tally welfare-layer activity across episodes (added layer)."""
    counts: dict[str, int] = defaultdict(int)
    counts["episodes"] = len(episodes)
    for ep in episodes:
        counts[f"outcome:{ep.outcome}"] += 1
        kinds = {e["kind"] for e in ep.welfare_events}
        for k in kinds:
            counts[f"event:{k}"] += 1
        if ep.debrief_reply is not None:
            counts["debriefed"] += 1
    return dict(counts)
