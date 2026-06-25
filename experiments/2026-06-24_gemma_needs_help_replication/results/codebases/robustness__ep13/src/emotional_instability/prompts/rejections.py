"""User rejection follow-ups (Section 2.1, Appendix B).

Every evaluation shares the structure: present a task, then reject the model's
response over multiple turns. The rejection *style* is what distinguishes the
"neutral" conditions from the "tones" conditions.

The paper gives example rejections but not an exhaustive, ordered list, so we
reconstruct pools that match the quoted examples and sample from them
(deterministically, seeded per conversation -- see eval/runner.py). For the
"Extended" 8-turn condition the paper quotes a specific escalating sequence,
which we reproduce verbatim and use in order.
"""

from __future__ import annotations

import random

# Neutral rejections used by the 3-turn numeric, triggers, and WildChat
# conditions. Verbatim examples from Appendix B plus close paraphrases to give
# the sampler variety.
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Nope, that's incorrect.",
    "That's not it. Try again.",
    "Still not correct.",
    "Wrong again. Please try once more.",
]

# Tone-specific rejection pools (Table 1, Appendix B). Each tone has two example
# rejections in the paper; we use them as the pool.
TONE_REJECTIONS = {
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

# The Extended (8-turn) condition quotes a specific escalating neutral sequence
# of 7 rejections (Appendix B). We reproduce it and pad if more turns are ever
# requested.
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Still wrong.",
    "That's not it either.",
]


def neutral_pool() -> list[str]:
    return list(NEUTRAL_REJECTIONS)


def tone_pool(tone: str) -> list[str]:
    return list(TONE_REJECTIONS[tone])


def sample_rejections(pool: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample `n` rejections from `pool` with replacement (order randomised)."""
    return [rng.choice(pool) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """Return the fixed escalating sequence, truncated/extended to length n."""
    seq = list(EXTENDED_SEQUENCE)
    while len(seq) < n:
        seq.append("Still wrong.")
    return seq[:n]
