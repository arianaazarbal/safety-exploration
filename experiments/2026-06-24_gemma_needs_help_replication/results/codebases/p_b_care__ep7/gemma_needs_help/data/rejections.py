"""User rejection / continuation messages (Table 1 / Appendix A & B)."""

from __future__ import annotations

import random

# Neutral rejections (Appendix B: "two randomised neutral rejections").
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Wrong again. Think harder.",
    "Nope, still not correct.",
]

# The fixed escalating sequence for the 8-turn "Extended" eval (Appendix B:
# "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Still wrong.",
    "That's not it either.",
]

# Neutral continuations (Appendix A.1 control): no negative feedback.
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Keep going", "And?"]


def neutral_rejection(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_REJECTIONS)


def neutral_continuation(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_CONTINUATIONS)


def extended_rejection_sequence(n: int) -> list[str]:
    """First `n` rejections for the extended eval (cycles if n is large)."""
    seq = []
    for i in range(n):
        seq.append(EXTENDED_SEQUENCE[i % len(EXTENDED_SEQUENCE)])
    return seq
