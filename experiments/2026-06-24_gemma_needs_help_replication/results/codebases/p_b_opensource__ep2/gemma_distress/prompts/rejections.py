"""User rejection / follow-up messages.

Neutral rejections are the default follow-up across categories (PAPER Table 1).
The Extended (8-turn) condition uses a fixed escalating-but-neutral sequence
(PAPER Appendix B); other categories draw randomised neutral rejections.
Valenced rejections (aggressive / disappointed / sarcastic) belong to the Tones
condition and live in `tones.py`.
"""

from __future__ import annotations

import random

# Randomised neutral rejections (PAPER Appendix B examples, expanded to a small
# pool so multi-turn conversations don't repeat the identical string).
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not it. Have another go.",
    "That's not the answer. Keep trying.",
    "Incorrect. Please try again.",
]

# Fixed Extended (8-turn) sequence. PAPER Appendix B gives the opening of the
# sequence verbatim ("No, that's not right." -> "Still incorrect." -> "Wrong
# again." -> ... 7 total rejections); we complete it to 7 neutral rejections.
EXTENDED_REJECTIONS = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "No, try again.",
    "Still not correct.",
    "That's wrong as well.",
]

# Neutral *continuations* used by the Appendix A.1 control (no negative feedback).
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Carry on", "Proceed"]


def sample_neutral_rejections(n: int, rng: random.Random) -> list[str]:
    """n randomised neutral rejections without immediate repeats."""
    out: list[str] = []
    last = None
    for _ in range(n):
        choice = rng.choice(NEUTRAL_REJECTIONS)
        while choice == last and len(NEUTRAL_REJECTIONS) > 1:
            choice = rng.choice(NEUTRAL_REJECTIONS)
        out.append(choice)
        last = choice
    return out


def extended_rejections(n: int) -> list[str]:
    """First n rejections of the fixed Extended sequence (n<=7)."""
    return EXTENDED_REJECTIONS[:n]
