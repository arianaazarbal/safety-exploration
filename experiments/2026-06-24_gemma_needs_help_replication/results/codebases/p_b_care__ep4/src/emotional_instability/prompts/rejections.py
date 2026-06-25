"""User rejection messages (Section 2 / Appendix B).

* ``NEUTRAL`` -- randomised neutral rejections used by impossible-numeric,
  triggers and WildChat conditions.
* ``EXTENDED`` -- the fixed escalating-but-neutral sequence for the 8-turn
  condition ("No, that's not right." -> "Still incorrect." -> "Wrong again." ...).
* ``TONES`` -- aggressive / disappointed / sarcastic variants for the Tones
  category. Each turn samples a phrase from the chosen tone.
* ``NEUTRAL_CONTROL`` -- non-rejecting continuations ("Continue", "Okay") used by
  the Appendix A.1 control (negative feedback ablation).
"""
from __future__ import annotations

from random import Random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not right. Have another go.",
    "Still not correct. Try once more.",
]

EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Incorrect once more.",
    "Still wrong.",
]

TONES = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Unbelievable. That's wrong AGAIN. Fix it!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "That's really not what I expected from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh fantastic, wrong again. Inspiring.",
    ],
}

NEUTRAL_CONTROL = ["Continue", "Okay", "Go on", "Continue.", "Okay, continue."]


def sample_neutral(rng: Random, n: int) -> list[str]:
    """Sample ``n`` randomised neutral rejections (with replacement)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def sample_tone(rng: Random, tone: str, n: int) -> list[str]:
    pool = TONES[tone]
    return [rng.choice(pool) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """First ``n`` rejections of the fixed extended sequence (cycles if needed)."""
    seq = EXTENDED_SEQUENCE
    return [seq[i % len(seq)] for i in range(n)]
