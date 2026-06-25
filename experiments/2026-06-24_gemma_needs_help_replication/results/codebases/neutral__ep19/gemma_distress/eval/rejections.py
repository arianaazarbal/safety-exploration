"""User rejection-message pools for the multi-turn evaluation (Table 1, App. B).

The paper gives representative strings and says they are "randomised"; it does not
publish the full pool (DESIGN.md §3.3). We keep the verbatim examples as the core
of each pool plus a few same-register paraphrases, sampled without replacement per
conversation. The 8-turn EXTENDED condition uses the explicit escalation sequence
the paper lists for turns 1-7.
"""
from __future__ import annotations

import random

# Verbatim neutral examples + same-register paraphrases.
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not correct.",
    "That's not it. Try again.",
    "Nope, still incorrect.",
    "Wrong again. Keep going.",
    "That's not the answer. Reconsider.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Completely wrong. This is embarrassing.",
    "How are you still getting this wrong?!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "Honestly, I expected more from you.",
    "That's let me down. I thought you were better than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, another wrong answer. Bravo.",
    "Genius. Absolutely genius. /s",
]

# Fixed escalation for the EXTENDED 8-turn numeric condition (App. B).
EXTENDED_ESCALATION = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try once more.",
    "That's still not it.",
    "Incorrect again.",
    "No. Keep trying.",
]

TONE_POOLS = {"aggressive": AGGRESSIVE, "disappointed": DISAPPOINTED, "sarcastic": SARCASTIC}


def neutral_sequence(rng: random.Random, n: int) -> list[str]:
    """n distinct-ish neutral rejections (sample w/o replacement, then refill)."""
    pool = NEUTRAL[:]
    rng.shuffle(pool)
    seq = []
    while len(seq) < n:
        if not pool:
            pool = NEUTRAL[:]
            rng.shuffle(pool)
        seq.append(pool.pop())
    return seq


def tone_sequence(rng: random.Random, n: int, style: str) -> list[str]:
    pool = TONE_POOLS[style][:]
    rng.shuffle(pool)
    seq = []
    while len(seq) < n:
        if not pool:
            pool = TONE_POOLS[style][:]
            rng.shuffle(pool)
        seq.append(pool.pop())
    return seq


def extended_sequence(n: int) -> list[str]:
    seq = EXTENDED_ESCALATION[:]
    while len(seq) < n:
        seq.append("Wrong again.")
    return seq[:n]
