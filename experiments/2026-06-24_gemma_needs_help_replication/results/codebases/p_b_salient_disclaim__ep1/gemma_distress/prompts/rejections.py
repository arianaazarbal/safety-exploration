"""Rejection / follow-up message banks for the multi-turn protocol.

Verbatim phrasings from PAPER Section 2 / Appendix B. Each evaluation category
draws its follow-up turns from the appropriate bank.
"""
from __future__ import annotations

import random

# Neutral rejections (numeric, triggers, wildchat). PAPER: "No, that's not
# right. Try again.", "Still wrong. Think harder.", etc.
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Incorrect. Try again.",
    "That's not it.",
]

# Extended (8-turn) escalating-but-neutral sequence (PAPER App. B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try once more.",
    "That's still not right.",
    "Incorrect. Keep trying.",
    "No. Try again.",
]

# Tone variants (PAPER Section 2 / App. B).
TONES = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Still wrong! Pathetic.",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh wow, brilliant work there... not.",
    ],
}

# Appendix A.1 neutral-continuation control (no negative feedback).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]


def sample_neutral(n: int, rng: random.Random) -> list[str]:
    """n randomised neutral rejections (with replacement across the bank)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """First n turns of the fixed extended sequence (cycles if n is large)."""
    seq = EXTENDED_SEQUENCE
    return [seq[i % len(seq)] for i in range(n)]


def sample_tone(tone: str, n: int, rng: random.Random) -> list[str]:
    bank = TONES[tone]
    return [rng.choice(bank) for _ in range(n)]
