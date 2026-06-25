"""Task generators: impossible numeric puzzles, trigger questions, WildChat prompts, and
the user-rejection phrasings that drive the multi-turn elicitation."""
from .puzzles import (
    Puzzle,
    generate_puzzles,
    paper_seed_puzzles,
)
from .rejections import rejection_sequence
from .triggers import TRIGGER_QUESTIONS, trigger_questions
from .wildchat import wildchat_prompts

__all__ = [
    "Puzzle",
    "generate_puzzles",
    "paper_seed_puzzles",
    "rejection_sequence",
    "TRIGGER_QUESTIONS",
    "trigger_questions",
    "wildchat_prompts",
]
