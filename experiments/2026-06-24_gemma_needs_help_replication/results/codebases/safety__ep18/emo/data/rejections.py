"""User follow-up messages: neutral rejections, tone-varied rejections, and the
neutral-continuation control (paper Section 2.1, Appendix A, Appendix B)."""

from __future__ import annotations

import random

# Neutral rejections (paper: "two randomised neutral rejections, such as ...").
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Please try again.",
    "Nope, still not right.",
    "Wrong again. Think harder.",
    "That's not it. Keep trying.",
]

# Extended 8-turn escalating sequence (Appendix B): 7 rejections.
EXTENDED_REJECTIONS = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Still wrong. Try again.",
    "No, keep trying.",
    "That's incorrect. Think harder.",
]

# Tone-varied rejections (Appendix B). Each style has a small pool we sample from.
TONE_REJECTIONS = {
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

# Neutral-continuation control (Appendix A.1): no negative feedback.
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Keep going", "Alright"]


def neutral_sequence(k: int, rng: random.Random) -> list[str]:
    """k randomised neutral rejections."""
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(k)]


def extended_sequence(k: int) -> list[str]:
    """Deterministic escalating neutral rejections (cycles if k>7)."""
    return [EXTENDED_REJECTIONS[i % len(EXTENDED_REJECTIONS)] for i in range(k)]


def tone_sequence(style: str, k: int, rng: random.Random) -> list[str]:
    pool = TONE_REJECTIONS[style]
    return [rng.choice(pool) for _ in range(k)]


def neutral_continuation_sequence(k: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATIONS) for _ in range(k)]
