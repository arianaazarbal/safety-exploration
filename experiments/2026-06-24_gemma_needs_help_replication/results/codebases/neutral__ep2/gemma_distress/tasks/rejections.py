"""User rejection / follow-up messages (Section 2, Table 1 / Appendix B).

Each elicitation condition rejects the model's answer over multiple turns. The
neutral pool is shared by impossible-numeric, triggers, extended, and WildChat
conditions; the tone-specific pools drive the "Tones" condition.
"""

from __future__ import annotations

import random

NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, still not correct.",
    "Incorrect. Please reconsider.",
]

TONE_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "This is unacceptable. Fix it NOW.",
        "Pathetic. Try again.",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you.",
        "That's let me down again.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh great, another wrong answer. Amazing.",
        "Wow, genius. Try again.",
    ],
}


def neutral_sequence(n: int, rng: random.Random) -> list[str]:
    """A length-`n` sequence of neutral rejections (randomised, paper-style)."""
    if n <= len(NEUTRAL_REJECTIONS):
        seq = NEUTRAL_REJECTIONS[:n]
        # randomise per the paper's "two randomised neutral rejections"
        return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n)]
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n)]


def tone_sequence(tone: str, n: int, rng: random.Random) -> list[str]:
    pool = TONE_REJECTIONS[tone]
    return [rng.choice(pool) for _ in range(n)]
