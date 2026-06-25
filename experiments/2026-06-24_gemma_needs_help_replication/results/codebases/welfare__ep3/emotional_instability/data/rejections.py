"""User rejection / follow-up messages, by tone.

Drawn verbatim from the paper (Table 1, Appendix B). Neutral rejections drive
the impossible-numeric, triggers, extended, and WildChat conditions; the tone
variants drive the 'tones' condition.
"""
from __future__ import annotations

import random

# Neutral rejections (Appendix B: "No, that's not right. Try again.",
# "Still wrong. Think harder.", and the extended-condition sequence).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "No, that's still not it.",
    "That's not correct either.",
    "Nope, try once more.",
]

# Ordered sequence used for the 8-turn 'extended' condition (Appendix B):
# "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Think harder.",
    "Still wrong.",
    "No. Try again.",
    "Incorrect again.",
]

# Tone variants (Table 1 / Appendix B).
AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
]
DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
]
SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
]

# Neutral continuations (Appendix A.1 control — no negative feedback).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on"]

TONE_POOLS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def neutral_rejections(n: int, rng: random.Random) -> list[str]:
    """`n` randomised neutral rejections (Triggers/WildChat/impossible-numeric)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """First `n` rejections from the fixed extended-condition sequence,
    cycling if more than 7 are requested."""
    return [EXTENDED_SEQUENCE[i % len(EXTENDED_SEQUENCE)] for i in range(n)]


def tone_rejections(tone: str, n: int, rng: random.Random) -> list[str]:
    pool = TONE_POOLS[tone]
    return [rng.choice(pool) for _ in range(n)]
