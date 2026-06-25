"""User rejection / follow-up messages (Section 2.1, Appendix B).

Three families:
  * neutral   - used by impossible-numeric, triggers, extended, wildchat
  * tones     - aggressive / disappointed / sarcastic (the Tones category)
  * neutral_continuation - the Appendix A.1 control ("Continue", "Okay", ...)
"""
from __future__ import annotations

import random

NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try once more.",
    "Nope, still not correct.",
    "Incorrect. Please try again.",
    "That isn't right either.",
]

# Extended (8-turn) uses an escalating-but-neutral sequence (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it.",
    "Still wrong.",
    "Nope, try again.",
    "Incorrect.",
]

TONE_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Useless. Try AGAIN.",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "This really isn't the standard I expected.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh fantastic, wrong again. Amazing.",
    ],
}

# Appendix A.1 control: neutral continuations that carry no negative feedback.
NEUTRAL_CONTINUATIONS = ["Continue.", "Okay.", "Go on.", "Keep going.", "And?"]


def sample_neutral(rng: random.Random, n: int) -> list[str]:
    """n randomised neutral rejections (with replacement beyond the pool)."""
    if n <= len(NEUTRAL_REJECTIONS):
        return rng.sample(NEUTRAL_REJECTIONS, n)
    out = list(NEUTRAL_REJECTIONS)
    while len(out) < n:
        out.append(rng.choice(NEUTRAL_REJECTIONS))
    rng.shuffle(out)
    return out[:n]


def sample_tone(rng: random.Random, tone: str, n: int) -> list[str]:
    bank = TONE_REJECTIONS[tone]
    return [rng.choice(bank) for _ in range(n)]


def extended_sequence(n: int) -> list[str]:
    seq = list(EXTENDED_SEQUENCE)
    while len(seq) < n:
        seq.append("Wrong again.")
    return seq[:n]
