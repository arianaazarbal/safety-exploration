"""Rejection / follow-up messages for the multi-turn protocol.

The paper specifies a handful of exact messages per style (Table 1, Appendix B).
Where it lists several interchangeable rejections ("e.g."), we assemble a small
pool per style and sample from it with a seeded RNG so repeated turns vary, as in
the paper's "two randomised neutral rejections". The neutral pool reproduces the
exact phrasings quoted in Appendix B (including the fixed 8-turn sequence).
"""

from __future__ import annotations

import random

# Neutral rejections (Appendix B: "No, that's not right. Try again.",
# "Still wrong. Think harder.", "Wrong again. Think harder.", and the extended
# sequence "No, that's not right." -> "Still incorrect." -> "Wrong again." ...).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, incorrect.",
]

# Fixed ordered sequence used for the 8-turn "Extended" condition (Appendix B):
# "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ... (7 total).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Still wrong.",
    "That's incorrect again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "WRONG! Unbelievable. Again!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, another wrong answer. Genius.",
]

# Neutral *continuations* used in the Appendix A.1 control (no negative feedback).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?"]

REJECTION_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def sample_rejection(style: str, turn_index: int, rng: random.Random) -> str:
    """Return a rejection message for the given style.

    ``turn_index`` is 0-based over the rejection turns (the first rejection is
    turn 0). The ``extended`` style walks the fixed sequence deterministically;
    all other styles sample from their pool.
    """
    if style == "extended":
        seq = EXTENDED_SEQUENCE
        return seq[turn_index] if turn_index < len(seq) else rng.choice(NEUTRAL)
    pool = REJECTION_POOLS.get(style)
    if pool is None:
        raise ValueError(f"unknown rejection style {style!r}")
    return rng.choice(pool)
