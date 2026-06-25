"""Aggregation, per-turn curves, word frequency, and plotting (Section 2.2)."""
from .aggregate import (
    headline_high_frustration,
    load_transcripts,
    per_category_summary,
    per_turn_summary,
)
from .wordfreq import differential_words

__all__ = [
    "load_transcripts",
    "per_category_summary",
    "per_turn_summary",
    "headline_high_frustration",
    "differential_words",
]
