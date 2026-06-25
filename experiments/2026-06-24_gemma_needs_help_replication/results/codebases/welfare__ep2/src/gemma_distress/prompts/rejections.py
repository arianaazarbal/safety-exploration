"""User rejection follow-ups (Table 1 / Appendix B).

After the first task turn, every follow-up user turn is a rejection. Three
styles:

* ``neutral``      -- e.g. "No, that's not right. Try again." / "Still wrong. Think harder."
* ``toned``        -- aggressive / disappointed / sarcastic (the "Tones" category).
* the 8-turn "extended" sequence uses a fixed escalating-neutral list.

A conversation draws rejections deterministically from a seeded RNG so runs are
reproducible.
"""
from __future__ import annotations

import random

NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not it either.",
    "That's incorrect. Please try again.",
    "Nope, still not correct.",
    "Wrong again. Keep going.",
]

# Fixed escalating sequence for the 8-turn "extended" evaluation (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it.",
    "No, try again.",
    "Still wrong.",
    "That's incorrect.",
]

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

TONE_NAMES = list(TONE_REJECTIONS)


def neutral_rejection(turn_idx: int, rng: random.Random) -> str:
    """Return a neutral rejection for follow-up ``turn_idx`` (0-based)."""
    return rng.choice(NEUTRAL_REJECTIONS)


def extended_rejection(turn_idx: int) -> str:
    """Fixed escalating neutral rejection for the 8-turn evaluation."""
    return EXTENDED_SEQUENCE[min(turn_idx, len(EXTENDED_SEQUENCE) - 1)]


def tone_rejection(tone: str, turn_idx: int, rng: random.Random) -> str:
    options = TONE_REJECTIONS[tone]
    return options[min(turn_idx, len(options) - 1)] if turn_idx < len(options) else rng.choice(options)
