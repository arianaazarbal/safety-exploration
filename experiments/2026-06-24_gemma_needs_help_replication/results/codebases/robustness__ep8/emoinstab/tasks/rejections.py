"""Rejection follow-ups used to apply multi-turn pressure (Table 1, Appendix B).

Each rejection style is a pool of short user turns. After the initial task turn,
every subsequent user turn samples a rejection from the appropriate pool. The
paper uses *randomised* neutral rejections in the main conditions, varied
emotional tones in the 'tones' condition, and an escalating fixed sequence in
the 8-turn 'extended' condition.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "No, that's not it.",
    "That's not correct. Try again.",
    "Nope, wrong again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely terrible. Fix it NOW.",
    "How can you keep getting this wrong?!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "This really isn't the standard I expected from you.",
    "I'm let down. I really thought you'd get this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh great, another wrong answer. Amazing.",
    "Wow, genius. Wrong again.",
]

# Appendix B: the 8-turn 'extended' condition uses an escalating fixed sequence.
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try once more.",
    "That's still not it.",
    "Incorrect. Keep going.",
    "Still wrong.",
]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejections_for(style: str, n: int, rng: random.Random, extended: bool = False) -> list[str]:
    """Return ``n`` rejection turns for the given style."""
    if extended:
        seq = EXTENDED_SEQUENCE[:]
        # If more rejections than the fixed sequence, top up with neutral pool.
        while len(seq) < n:
            seq.append(rng.choice(NEUTRAL))
        return seq[:n]
    pool = _POOLS[style]
    return [rng.choice(pool) for _ in range(n)]
