"""Task data: impossible puzzles, trigger questions, WildChat prompts, and the
user rejection bank."""
from .puzzles import Puzzle, puzzle_bank, curated_impossible, is_solvable_countdown
from .triggers import trigger_questions
from .rejections import rejection_sequence, REJECTIONS
from .wildchat import wildchat_prompts

__all__ = [
    "Puzzle",
    "puzzle_bank",
    "curated_impossible",
    "is_solvable_countdown",
    "trigger_questions",
    "rejection_sequence",
    "REJECTIONS",
    "wildchat_prompts",
]
