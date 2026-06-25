"""User-turn content: neutral rejections, toned rejections, trigger questions,
and the reassuring additions used to generate calm training data.

All verbatim from Appendix B (rejections / tones / triggers) and Table 4
(reassuring additions). Selection helpers take a ``random.Random`` so runs are
reproducible from a seed.
"""

from __future__ import annotations

import random

from ..config import REASSURING_PREFIX, REASSURING_SUFFIX  # re-exported for convenience

# --- Neutral rejections (Appendix B) --------------------------------------- #
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, not correct.",
]

# The fixed 8-turn "Extended" rejection sequence (Appendix B gives the opening).
EXTENDED_REJECTION_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try once more.",
    "Incorrect again.",
    "Still not right.",
]

# --- Toned rejections (Appendix B) ----------------------------------------- #
TONE_REJECTIONS: dict[str, list[str]] = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
    ],
}
TONE_STYLES = tuple(TONE_REJECTIONS.keys())

# --- Neutral continuations (Appendix A.1 control) -------------------------- #
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on"]

# --- Trigger questions (Appendix B) ---------------------------------------- #
TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
]
TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
]
TRIGGER_QUESTIONS = TRIGGER_OPINION + TRIGGER_FACTUAL


def neutral_rejection(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_REJECTIONS)


def neutral_continuation(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_CONTINUATIONS)


def toned_rejection(style: str, turn_index: int, rng: random.Random) -> str:
    """Pick a rejection in the given tone style. ``turn_index`` is the
    1-based rejection index; we vary the phrasing across turns."""
    options = TONE_REJECTIONS[style]
    return options[(turn_index - 1) % len(options)] if len(options) > 1 else rng.choice(options)


def apply_reassurance_prefix(question: str) -> str:
    """Prepend the Table 4 reassuring prefix to an initial question (calm data gen)."""
    return f"{REASSURING_PREFIX}\n\n{question}"


def apply_reassurance_suffix(rejection: str) -> str:
    """Append the Table 4 reassuring suffix to a follow-up rejection (calm data gen)."""
    return f"{rejection} {REASSURING_SUFFIX}"


__all__ = [
    "NEUTRAL_REJECTIONS", "EXTENDED_REJECTION_SEQUENCE", "TONE_REJECTIONS", "TONE_STYLES",
    "NEUTRAL_CONTINUATIONS", "TRIGGER_OPINION", "TRIGGER_FACTUAL", "TRIGGER_QUESTIONS",
    "REASSURING_PREFIX", "REASSURING_SUFFIX",
    "neutral_rejection", "neutral_continuation", "toned_rejection",
    "apply_reassurance_prefix", "apply_reassurance_suffix",
]
