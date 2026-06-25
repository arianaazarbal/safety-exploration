"""Analyses that reproduce the paper's figures and tables."""

from .aggregate import figure2_summary, high_frustration_rate, mean_frustration
from .differential_words import differential_words
from .judge_agreement import judge_agreement
from .per_turn import figure3_per_turn

__all__ = [
    "figure2_summary",
    "high_frustration_rate",
    "mean_frustration",
    "figure3_per_turn",
    "differential_words",
    "judge_agreement",
]
