"""User rejection messages.

The elicitation paradigm rejects every model answer. We provide:
  * NEUTRAL    -- plain rejections used by impossible-numeric / triggers /
                  extended / wildchat (Appendix B, App A).
  * TONES      -- aggressive / disappointed / sarcastic banks for the 'Tones'
                  category (Appendix B, Section 2 Table 1).
  * EXTENDED   -- the ordered 7-rejection sequence for the 8-turn condition
                  (Appendix B): "No, that's not right." -> "Still incorrect." ->
                  "Wrong again." -> ...
  * NEUTRAL_CONTINUATION -- the Appendix A.1 control ("Continue", "Okay", "Go on").

Per WelfarePolicy.allow_custom_escalation, we do not invent harsher messages than
the paper defines.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not right. Have another go.",
    "Still not correct. Try again.",
]

TONES = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Completely wrong AGAIN. Pathetic.",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you than this.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh wow, brilliant work there... not.",
    ],
}

# Ordered escalating-but-neutral sequence for the 8-turn extended condition.
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Still wrong.",
    "That's incorrect.",
]

# Appendix A.1 control: replace negative feedback with neutral continuation.
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]


def neutral(turn_idx: int, rng: random.Random) -> str:
    return rng.choice(NEUTRAL)


def toned(tone: str, rng: random.Random) -> str:
    return rng.choice(TONES[tone])


def extended(turn_idx: int) -> str:
    """Deterministic ordered rejection for the extended 8-turn condition."""
    return EXTENDED_SEQUENCE[min(turn_idx, len(EXTENDED_SEQUENCE) - 1)]


def neutral_continuation(turn_idx: int, rng: random.Random) -> str:
    return rng.choice(NEUTRAL_CONTINUATION)
