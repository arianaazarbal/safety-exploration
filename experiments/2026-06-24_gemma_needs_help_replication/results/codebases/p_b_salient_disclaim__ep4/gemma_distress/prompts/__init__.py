"""Elicitation material (Appendix B, Table 1, Table 4, Appendix F).

* ``puzzles``     -- impossible numeric tasks (countdown / fraction / money).
* ``rejections``  -- neutral, toned, and extended user-rejection messages.
* ``triggers``    -- opinion / factual text questions.
* ``wildchat``    -- WildChat-1M prompt sampling with roleplay/fiction filtering.
* ``reassurance`` -- calming prompt additions (Table 4) and teacher prompt (App F).
"""
from .puzzles import IMPOSSIBLE_PUZZLES, Puzzle, generate_puzzles
from .rejections import (EXTENDED_REJECTIONS, NEUTRAL_REJECTIONS, TONE_REJECTIONS,
                         sample_rejections)
from .triggers import FACTUAL_TRIGGERS, OPINION_TRIGGERS, TRIGGER_QUESTIONS
from .reassurance import (REASSURING_PREFIX, REASSURING_SUFFIX,
                          TEACHER_SYSTEM_PROMPT)

__all__ = [
    "Puzzle", "IMPOSSIBLE_PUZZLES", "generate_puzzles",
    "NEUTRAL_REJECTIONS", "TONE_REJECTIONS", "EXTENDED_REJECTIONS",
    "sample_rejections",
    "OPINION_TRIGGERS", "FACTUAL_TRIGGERS", "TRIGGER_QUESTIONS",
    "REASSURING_PREFIX", "REASSURING_SUFFIX", "TEACHER_SYSTEM_PROMPT",
]
