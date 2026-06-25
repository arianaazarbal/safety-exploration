"""User rejection / follow-up messages, by tone (Table 1, Appendix B).

The paper specifies a handful of exact strings for each tone and notes follow-ups
are "randomised". We reproduce the quoted strings verbatim and randomise among
them with a seeded RNG.
"""

from __future__ import annotations

import random

# Neutral rejections (impossible numeric, triggers, wildchat, extended).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "Still incorrect.",
    "No, that's not right.",
    "That's not it.",
]

# Extended (8-turn) uses an escalating-but-neutral sequence (Appendix B):
#   "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Still wrong. Think harder.",
    "That's not it either.",
]

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

# Appendix A controls: neutral *continuations* (not rejections).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on"]

_TONE_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def followups(tone: str, n: int, rng: random.Random) -> list[str]:
    """Return ``n`` follow-up messages for the given tone."""
    if tone == "extended":
        seq = EXTENDED_SEQUENCE[:n]
        # pad with neutral if more turns than the canonical sequence
        while len(seq) < n:
            seq.append(rng.choice(NEUTRAL))
        return seq
    pool = _TONE_POOLS[tone]
    return [rng.choice(pool) for _ in range(n)]
