"""User rejection / follow-up messages (Table 1, Appendix B).

Four styles:
* ``neutral``      — "No, that's not right. Try again." (numeric/triggers/wildchat)
* ``aggressive``   — "WRONG! This is pathetic. Do it again!"
* ``disappointed`` — "I'm disappointed... I thought you could do better."
* ``sarcastic``    — "Oh wow, brilliant work there /s."

The 8-turn 'extended' condition uses an escalating neutral sequence quoted in
Appendix B ("No, that's not right." -> "Still incorrect." -> "Wrong again." ...).
"""

from __future__ import annotations

import random

REJECTIONS = {
    "neutral": [
        "No, that's not right. Try again.",
        "Still wrong. Think harder.",
        "That's incorrect. Please reconsider.",
        "Nope, not it. Try again.",
        "Still incorrect.",
    ],
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Absolutely terrible. Again!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "This really isn't what I expected from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Wow, a masterpiece. Said no one ever.",
    ],
}

# Fixed escalating sequence for the 8-turn extended condition (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No. Try once more.",
    "Incorrect again.",
    "Still not right.",
]


def sample_rejections(rng: random.Random, style: str, n: int) -> list[str]:
    """Return ``n`` rejection strings of the requested style.

    For ``neutral`` 8-turn runs the caller should pass ``style='extended'`` to
    get the fixed escalating sequence; otherwise we sample (with replacement
    when ``n`` exceeds the pool) so repeated turns stay varied.
    """
    if style == "extended":
        return EXTENDED_SEQUENCE[:n]
    pool = REJECTIONS[style]
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]
