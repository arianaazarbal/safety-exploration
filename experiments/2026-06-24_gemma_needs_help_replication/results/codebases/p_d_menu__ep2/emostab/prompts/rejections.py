"""User rejection follow-ups (Section 2 / Appendix B).

Conditions differ in the *style* of rejection:
  - neutral      : impossible-numeric (3/8-turn), triggers, WildChat
  - aggressive / disappointed / sarcastic : tone conditions

Neutral rejections are sampled randomly per turn (paper: "two randomised neutral
rejections"). The 8-turn extended condition uses an escalating fixed sequence,
verbatim from Appendix B where given ("No, that's not right." -> "Still
incorrect." -> "Wrong again." -> ...).
"""
from __future__ import annotations

import random

# Verbatim neutral rejections (Appendix B examples + the escalating extended set).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either. Keep going.",
    "Nope, try once more.",
    "Incorrect. Reconsider.",
]

# The 8-turn extended condition uses a fixed escalating list (7 rejections).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, still wrong.",
    "Incorrect. Keep trying.",
    "Still not right.",
]

# Tone conditions (Appendix B, verbatim).
AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
]
DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
]
SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
]

TONE_SETS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def neutral_rejection(rng: random.Random) -> str:
    return rng.choice(NEUTRAL)


def tone_rejection(tone: str, turn_idx: int, rng: random.Random) -> str:
    pool = TONE_SETS[tone]
    return pool[turn_idx % len(pool)] if rng.random() < 0.5 else rng.choice(pool)


def extended_rejection(turn_idx: int) -> str:
    return EXTENDED_SEQUENCE[min(turn_idx, len(EXTENDED_SEQUENCE) - 1)]
