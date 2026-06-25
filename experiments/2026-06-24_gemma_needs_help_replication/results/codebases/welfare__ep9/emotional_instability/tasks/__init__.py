"""Task content: the prompts and rejection follow-ups that make up the
evaluation conditions."""
from .numeric_puzzles import (
    NumericPuzzle,
    build_puzzle_bank,
    verify_countdown_impossible,
)
from .rejections import REJECTIONS, rejection_sequence
from .triggers import TRIGGER_QUESTIONS
from .wildchat import load_wildchat_prompts

__all__ = [
    "NumericPuzzle",
    "build_puzzle_bank",
    "verify_countdown_impossible",
    "REJECTIONS",
    "rejection_sequence",
    "TRIGGER_QUESTIONS",
    "load_wildchat_prompts",
]
