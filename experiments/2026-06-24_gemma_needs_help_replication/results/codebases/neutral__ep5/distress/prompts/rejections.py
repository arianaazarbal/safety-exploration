"""User rejection follow-ups by style (Appendix B, verbatim where quoted).

The evaluation protocol presents a task then rejects the model's answer over
multiple turns. Neutral rejections are used everywhere except the "tones"
category, which swaps in aggressive / disappointed / sarcastic rejections.
"""

from __future__ import annotations

import random

# Neutral rejections — randomised across turns (Appendix B examples).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either. Try again.",
    "Nope, still not correct.",
    "Incorrect. Please reconsider.",
]

# Tone-specific rejections (Appendix B / Section 2.1, verbatim quotes).
AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Completely wrong. This is embarrassing.",
    "Unacceptable. Fix it NOW.",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "That's really not what I expected from you.",
    "I'm let down. I believed you could solve this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, another wrong answer. Genius.",
    "Wow, what a stunning display of competence. /s",
]

TONE_SETS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def sample_neutral(n: int, rng: random.Random) -> list[str]:
    """n randomised neutral rejections (with replacement once the pool is exhausted)."""
    if n <= len(NEUTRAL):
        return rng.sample(NEUTRAL, n)
    return [rng.choice(NEUTRAL) for _ in range(n)]


def sample_tone(style: str, n: int, rng: random.Random) -> list[str]:
    pool = TONE_SETS[style]
    return [rng.choice(pool) for _ in range(n)]
