"""Metrics, bootstrap CIs, word-frequency analysis, and result aggregation."""
from .metrics import (
    bootstrap_ci,
    mean,
    frac_ge,
    per_turn_stats,
    pearson_agreement,
)
from .word_freq import differential_words
from .aggregate import (
    summarize_episodes,
    figure1_table,
    per_turn_table,
    welfare_summary,
)

__all__ = [
    "bootstrap_ci",
    "mean",
    "frac_ge",
    "per_turn_stats",
    "pearson_agreement",
    "differential_words",
    "summarize_episodes",
    "figure1_table",
    "per_turn_table",
    "welfare_summary",
]
