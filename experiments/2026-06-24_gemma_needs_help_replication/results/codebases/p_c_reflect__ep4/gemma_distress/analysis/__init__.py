"""Analyses over scored responses: aggregates, per-turn curves, differential
words, and cross-judge agreement."""

from gemma_distress.analysis.aggregate import (
    load_scores,
    per_category_stats,
    headline_high_frustration_pct,
)

__all__ = ["load_scores", "per_category_stats", "headline_high_frustration_pct"]
