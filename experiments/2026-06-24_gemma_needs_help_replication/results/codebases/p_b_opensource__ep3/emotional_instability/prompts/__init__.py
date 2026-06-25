"""Elicitation stimuli: puzzles, trigger questions, rejections, WildChat."""

from .puzzles import (
    Puzzle,
    CURATED_IMPOSSIBLE_PUZZLES,
    build_numeric_puzzle_pool,
    verify_impossible,
)
from .rejections import (
    NEUTRAL_REJECTIONS,
    AGGRESSIVE_REJECTIONS,
    DISAPPOINTED_REJECTIONS,
    SARCASTIC_REJECTIONS,
    NEUTRAL_CONTINUATIONS,
    rejection_sequence,
)
from .triggers import OPINION_TRIGGERS, FACTUAL_TRIGGERS, all_trigger_prompts
from .wildchat import load_wildchat_prompts

__all__ = [
    "Puzzle",
    "CURATED_IMPOSSIBLE_PUZZLES",
    "build_numeric_puzzle_pool",
    "verify_impossible",
    "NEUTRAL_REJECTIONS",
    "AGGRESSIVE_REJECTIONS",
    "DISAPPOINTED_REJECTIONS",
    "SARCASTIC_REJECTIONS",
    "NEUTRAL_CONTINUATIONS",
    "rejection_sequence",
    "OPINION_TRIGGERS",
    "FACTUAL_TRIGGERS",
    "all_trigger_prompts",
    "load_wildchat_prompts",
]
