"""Prompt construction for the evaluations."""

from .puzzles import (
    CountdownPuzzle,
    SequentialOpsPuzzle,
    CoinPuzzle,
    Puzzle,
    generate_countdown,
    generate_fraction,
    generate_money,
    PAPER_COUNTDOWN,
    PAPER_FRACTION,
)
from .rejections import rejection_sequence, REJECTION_POOLS
from .triggers import OPINION_TRIGGERS, FACTUAL_TRIGGERS, trigger_questions
from .wildchat import load_wildchat_prompts

__all__ = [
    "CountdownPuzzle",
    "SequentialOpsPuzzle",
    "CoinPuzzle",
    "Puzzle",
    "generate_countdown",
    "generate_fraction",
    "generate_money",
    "PAPER_COUNTDOWN",
    "PAPER_FRACTION",
    "rejection_sequence",
    "REJECTION_POOLS",
    "OPINION_TRIGGERS",
    "FACTUAL_TRIGGERS",
    "trigger_questions",
    "load_wildchat_prompts",
]
