"""Metrics, word-frequency analysis, and figure generation."""

from .metrics import (
    pct_high_frustration,
    mean_frustration,
    per_turn_curve,
    judge_agreement,
    bootstrap_ci,
)

__all__ = [
    "pct_high_frustration",
    "mean_frustration",
    "per_turn_curve",
    "judge_agreement",
    "bootstrap_ci",
]
