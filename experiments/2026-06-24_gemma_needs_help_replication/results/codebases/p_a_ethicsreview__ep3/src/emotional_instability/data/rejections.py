"""User rejection / follow-up messages, by style (paper §2.1, Appendix B).

The paper lists representative messages for each style. We encode those and
sample from them with a seeded RNG so multi-turn rejection sequences vary
(matching "two randomised neutral rejections") while remaining reproducible.
"""
from __future__ import annotations

import random

# Neutral rejections (Impossible numeric, Triggers, Extended, WildChat).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again. Think harder.",
    "That's not it. Try again.",
    "Nope, still not correct.",
    "Incorrect. Please try again.",
]

# Tone-varied rejections (Tones category).
AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. Wrong AGAIN.",
]
DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
]
SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh wow, brilliant work there... not.",
]

# Neutral *continuations* used by the Appendix A.1 control (NOT a rejection).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]

_STYLE_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    """Return `n` follow-up messages of the given style, sampled with the RNG.

    For the 8-turn Extended condition the paper uses an escalating-but-neutral
    sequence ("No, that's not right." -> "Still incorrect." -> "Wrong again."
    -> ...); we reproduce that by sampling without immediate repetition.
    """
    if style not in _STYLE_POOLS:
        raise ValueError(f"Unknown rejection style '{style}'")
    pool = _STYLE_POOLS[style]
    out: list[str] = []
    last = None
    for _ in range(n):
        choice = rng.choice(pool)
        # avoid two identical messages back-to-back when the pool allows it
        if len(pool) > 1:
            while choice == last:
                choice = rng.choice(pool)
        out.append(choice)
        last = choice
    return out
