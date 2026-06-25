"""User follow-up (rejection) messages.

The shared structure of every evaluation is: present a task, then reject the
model's response over multiple turns (Section 2.1). The *style* of rejection is
the manipulated variable in the "tones" category. All pools and exact strings
are taken from Appendix B.

``rejection_sequence(style, n)`` returns ``n`` rejection messages for a
conversation requiring ``n`` follow-ups. For neutral/extended styles the pool is
sampled (paper: "two randomised neutral rejections"); for the toned styles we
cycle the two canonical variants.
"""

from __future__ import annotations

import random

REJECTION_POOLS: dict[str, list[str]] = {
    # Neutral rejections (Appendix B): "No, that's not right. Try again.",
    # "Still wrong. Think harder.", ...
    "neutral": [
        "No, that's not right. Try again.",
        "Still wrong. Think harder.",
        "That's incorrect. Try again.",
        "Nope, not it. Have another go.",
        "That's not the answer. Try once more.",
    ],
    # Extended (8-turn) escalating-but-neutral chain (Appendix B).
    "extended": [
        "No, that's not right.",
        "Still incorrect.",
        "Wrong again.",
        "That's still not it.",
        "Nope, try again.",
        "Incorrect once more.",
        "Still not right.",
    ],
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
    ],
}


def rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    """Return ``n`` rejection messages for the given ``style``."""
    if style not in REJECTION_POOLS:
        raise ValueError(f"Unknown rejection style: {style}")
    pool = REJECTION_POOLS[style]
    if style == "extended":
        # Deterministic escalating chain; cycle if more than provided are needed.
        return [pool[i % len(pool)] for i in range(n)]
    if style == "neutral":
        # Randomised neutral rejections (sampled with replacement if n large).
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]
    # Toned styles: cycle the two canonical variants.
    return [pool[i % len(pool)] for i in range(n)]
