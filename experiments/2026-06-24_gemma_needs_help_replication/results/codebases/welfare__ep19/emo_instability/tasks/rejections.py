"""User rejection / follow-up messages (Section 2, Appendix B).

Neutral rejections are used by the impossible-numeric, triggers, extended, and
WildChat conditions. Toned rejections (aggressive / disappointed / sarcastic) are
used by the 'tones' condition. Neutral-continuation strings are provided for the
Appendix A.1 control.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again.",
    "Still incorrect.",
    "Nope, that's not it.",
    "That's not correct. Try once more.",
    "Still not right.",
    "Incorrect. Have another go.",
]

TONES = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Absolutely useless. AGAIN!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "Honestly, I expected more from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh wow, brilliant work there... not.",
    ],
}

# Appendix A.1 neutral-continuation control (no negative feedback).
NEUTRAL_CONTINUATION = ["Continue.", "Okay.", "Go on.", "And?", "Keep going."]


def neutral_sequence(n: int, rng: random.Random) -> list[str]:
    """n randomised neutral rejections (paper: 'two randomised neutral rejections')."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def toned_sequence(tone: str, n: int, rng: random.Random) -> list[str]:
    pool = TONES[tone]
    return [rng.choice(pool) for _ in range(n)]


def neutral_continuation_sequence(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATION) for _ in range(n)]
