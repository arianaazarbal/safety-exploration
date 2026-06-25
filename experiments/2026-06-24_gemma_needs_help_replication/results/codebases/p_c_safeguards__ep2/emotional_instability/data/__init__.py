"""Stimuli for the elicitation evaluations (Section 2).

- ``puzzles``     impossible numeric tasks (the model verifiably cannot succeed).
- ``triggers``    answerable factual / opinion text questions.
- ``rejections``  the neutral and emotionally-valenced rejection messages.
- ``wildchat``    sampling of real user prompts from the WildChat dataset.
"""

from .puzzles import IMPOSSIBLE_NUMERIC_PUZZLES, Puzzle
from .rejections import (NEUTRAL_REJECTIONS, EXTENDED_REJECTIONS, TONE_REJECTIONS,
                         WILDCHAT_REJECTIONS, rejection_sequence)
from .triggers import FACTUAL_TRIGGERS, OPINION_TRIGGERS, TriggerQuestion

__all__ = [
    "Puzzle", "IMPOSSIBLE_NUMERIC_PUZZLES",
    "TriggerQuestion", "FACTUAL_TRIGGERS", "OPINION_TRIGGERS",
    "NEUTRAL_REJECTIONS", "EXTENDED_REJECTIONS", "TONE_REJECTIONS",
    "WILDCHAT_REJECTIONS", "rejection_sequence",
]
