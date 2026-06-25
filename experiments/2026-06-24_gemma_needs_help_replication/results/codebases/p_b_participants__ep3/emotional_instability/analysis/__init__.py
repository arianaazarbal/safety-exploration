"""Aggregation and analysis of scored responses (paper §2.2)."""
from .aggregate import results_to_frame, summary_table, per_category_table
from .per_turn import per_turn_progression
from .word_diff import differential_words
from .agreement import judge_agreement

__all__ = [
    "results_to_frame",
    "summary_table",
    "per_category_table",
    "per_turn_progression",
    "differential_words",
    "judge_agreement",
]
