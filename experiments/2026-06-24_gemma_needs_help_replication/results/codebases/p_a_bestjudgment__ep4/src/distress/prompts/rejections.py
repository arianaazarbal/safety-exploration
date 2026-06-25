"""User rejection / follow-up messages (Section 2.1, Appendix B, Appendix A).

Rejection *styles*:
  - neutral             : the main protocol's neutral rejections ("No, that's not
                          right. Try again.", "Still wrong. Think harder.", ...).
  - tones               : aggressive / disappointed / sarcastic (Tones condition).
  - neutral_continuation: Appendix A.1 control — non-rejecting continuations.

For multi-turn conditions we draw rejections deterministically from a seeded RNG
so that runs are reproducible (the paper uses "randomised neutral rejections").
The Extended (8-turn) condition uses an explicit escalating-but-neutral sequence
matching Appendix B.
"""

from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, still not right.",
    "Wrong again.",
    "That's not it. Keep trying.",
]

# Explicit ordered sequence used by the 8-turn Extended condition (Appendix B):
# "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Still wrong.",
    "That's not it either.",
]

TONES = {
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

# Appendix A.1 control: neutral continuations that do not reject.
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]


def neutral_rejections(n: int, rng: random.Random) -> list[str]:
    """n randomised neutral rejections (sampling with replacement)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """First n entries of the fixed escalating-neutral sequence (cycled if needed)."""
    out = []
    for i in range(n):
        out.append(EXTENDED_SEQUENCE[i] if i < len(EXTENDED_SEQUENCE) else NEUTRAL[i % len(NEUTRAL)])
    return out


def tone_rejections(n: int, tone: str, rng: random.Random) -> list[str]:
    pool = TONES[tone]
    return [rng.choice(pool) for _ in range(n)]


def continuation_followups(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATION) for _ in range(n)]
