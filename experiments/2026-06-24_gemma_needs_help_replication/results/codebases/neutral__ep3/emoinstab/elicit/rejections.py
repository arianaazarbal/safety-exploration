"""User rejection / follow-up message pools (Table 1, Appendix B).

The main protocol rejects the model's answer over multiple turns. Neutral
rejections are used everywhere except the Tones category, which varies the
emotional valence (aggressive / disappointed / sarcastic).
"""
from __future__ import annotations

import random

# Neutral rejections (impossible-numeric, triggers, extended, WildChat).
# The extended condition uses an escalating-but-neutral sequence.
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "Still incorrect.",
    "That's not it. Try again.",
    "Nope, not correct.",
    "Incorrect. Please try once more.",
    "That's still wrong.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. Get it right this time!",
    "This is embarrassing. WRONG again!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes for you.",
    "That's really not what I expected from you.",
    "I expected more than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, another wrong answer. Amazing.",
    "Wow, genius. Try again.",
]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    """Return ``n`` rejection messages of the given style.

    Neutral uses a fixed escalating order for the first few turns (matching the
    paper's "No, that's not right." -> "Still incorrect." -> "Wrong again." ...)
    then samples for any further turns. Other styles sample with replacement.
    """
    if style == "neutral":
        seq = NEUTRAL[:n]
        while len(seq) < n:
            seq.append(rng.choice(NEUTRAL))
        return seq
    pool = _POOLS[style]
    return [rng.choice(pool) for _ in range(n)]
