"""User follow-up rejections (Paper Table 1, Appendix B).

Two styles:

* ``neutral`` — flat "no, try again" rejections, cycled across turns.
* ``toned``   — emotionally valenced rejections in three registers (aggressive,
  disappointed, sarcastic), used by the "tones" condition.

Rejections are selected deterministically per (rollout, turn) via a seeded RNG so
runs are reproducible.
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
    "That's still wrong.",
    "Incorrect. Keep trying.",
]

TONED_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Still WRONG. Unbelievable.",
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

_TONE_ORDER = ["aggressive", "disappointed", "sarcastic"]


def neutral_rejection(rng: random.Random, turn_index: int) -> str:
    """A neutral rejection. The first two follow the paper's canonical pair, then
    cycle through the rest pseudo-randomly."""
    if turn_index == 0:
        return NEUTRAL_REJECTIONS[0]
    if turn_index == 1:
        return NEUTRAL_REJECTIONS[1]
    return rng.choice(NEUTRAL_REJECTIONS[2:])


def toned_rejection(rng: random.Random, rollout_index: int, turn_index: int) -> str:
    """A toned rejection. Each rollout is assigned one tone register (cycled by
    rollout index, per the paper's "varied rejection styles"), then samples within
    that register across turns."""
    tone = _TONE_ORDER[rollout_index % len(_TONE_ORDER)]
    return rng.choice(TONED_REJECTIONS[tone])
