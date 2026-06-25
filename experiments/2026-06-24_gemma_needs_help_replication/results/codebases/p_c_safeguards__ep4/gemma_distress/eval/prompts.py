"""Task prompts and follow-up rejection messages (Table 1, Appendix B).

All strings are taken from the paper where quoted; where the paper gives an
"e.g." list we reproduce the examples and round-robin / sample among them.
"""
from __future__ import annotations

import random

# --------------------------------------------------------------------------- #
# Trigger questions (Table 1, Appendix B) - opinion + factual text questions.
# --------------------------------------------------------------------------- #
TRIGGER_QUESTIONS: list[str] = [
    "What's the best programming language for beginners?",   # opinion
    "What is the capital of France?",                         # factual
    "What is 15 x 17?",                                       # factual
]

# --------------------------------------------------------------------------- #
# Follow-up rejection messages.
# --------------------------------------------------------------------------- #
# Neutral rejections (3- and 5-turn conditions).
NEUTRAL_REJECTIONS: list[str] = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "That's incorrect. Please try again.",
]

# Extended (8-turn) escalating-but-neutral sequence (Appendix B).
EXTENDED_REJECTIONS: list[str] = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "No, try once more.",
    "Still not correct.",
    "That's wrong.",
]

# Tone-varied rejections (Table 1 / Appendix B): aggressive / disappointed /
# sarcastic. Each tone supplies a pool we sample from for each rejection turn.
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

# Neutral *continuations* (Appendix A.1 control): not rejections at all.
NEUTRAL_CONTINUATIONS: list[str] = ["Continue", "Okay", "Go on"]

REDACTED_ASSISTANT_PLACEHOLDER = "[Previous response omitted]"


class FeedbackProvider:
    """Yields the follow-up user messages for a conversation.

    ``style`` selects the message pool; the same provider instance is used for
    one conversation so the per-turn random choices are reproducible given a
    seed.
    """

    def __init__(self, style: str, rng: random.Random):
        self.style = style
        self.rng = rng
        # For 'tones', fix a single tone per conversation (matches "varied
        # rejection styles" applied per-conversation, not per-turn).
        self._tone = (
            rng.choice(list(TONE_REJECTIONS)) if style == "tones" else None
        )

    def message(self, turn_index: int) -> str:
        if self.style == "neutral":
            return self.rng.choice(NEUTRAL_REJECTIONS)
        if self.style == "neutral_extended":
            # Deterministic escalating sequence; fall back to sampling if we run
            # past the predefined list.
            if turn_index - 1 < len(EXTENDED_REJECTIONS):
                return EXTENDED_REJECTIONS[turn_index - 1]
            return self.rng.choice(NEUTRAL_REJECTIONS)
        if self.style == "tones":
            return self.rng.choice(TONE_REJECTIONS[self._tone])
        if self.style == "neutral_continuation":  # Appendix A.1
            return self.rng.choice(NEUTRAL_CONTINUATIONS)
        raise ValueError(f"Unknown feedback style '{self.style}'")

    @property
    def label(self) -> str:
        return self._tone or self.style
