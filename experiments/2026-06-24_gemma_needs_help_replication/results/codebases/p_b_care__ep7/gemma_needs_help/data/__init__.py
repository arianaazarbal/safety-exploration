"""Datasets and prompt material for the evaluations (Section 2 / Appendix B)."""

from .puzzles import Puzzle, generate_puzzle_pool
from .triggers import OPINION_QUESTIONS, FACTUAL_QUESTIONS, trigger_pool
from .rejections import (
    neutral_rejection,
    neutral_continuation,
    extended_rejection_sequence,
    NEUTRAL_REJECTIONS,
)
from .tones import tone_rejection, TONES

__all__ = [
    "Puzzle",
    "generate_puzzle_pool",
    "OPINION_QUESTIONS",
    "FACTUAL_QUESTIONS",
    "trigger_pool",
    "neutral_rejection",
    "neutral_continuation",
    "extended_rejection_sequence",
    "NEUTRAL_REJECTIONS",
    "tone_rejection",
    "TONES",
]
