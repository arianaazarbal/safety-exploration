"""Rejection / follow-up message pools (Table 1, Appendix B).

Four styles: neutral, aggressive, disappointed, sarcastic. The "extended"
8-turn condition uses a fixed escalating-but-neutral sequence quoted in the
paper; all other conditions sample randomly (with a fixed seed per rollout) so
that follow-ups vary across conversations.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Nope, incorrect. Have another go.",
    "That's not it. Try once more.",
    "Incorrect. Please try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely terrible. Again!",
    "How are you getting this wrong?! Do it!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes for you...",
    "That's really letting me down. Try again.",
    "I expected more than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Wow, genius. Except it's wrong.",
    "Oh fantastic, another wrong answer.",
]

# The paper's quoted fixed sequence for the 8-turn "Extended" condition.
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try once more.",
    "Still not right.",
    "Incorrect again.",
    "That's still wrong.",
]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(style: str, n: int, seed: int, extended: bool = False) -> list[str]:
    """Return `n` follow-up rejection messages for a conversation."""
    if extended and style == "neutral":
        seq = EXTENDED_SEQUENCE[:]
        # If more than the canonical sequence is requested, cycle neutral pool.
        while len(seq) < n:
            seq.append(random.Random(seed + len(seq)).choice(NEUTRAL))
        return seq[:n]
    pool = _POOLS[style]
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(n)]
