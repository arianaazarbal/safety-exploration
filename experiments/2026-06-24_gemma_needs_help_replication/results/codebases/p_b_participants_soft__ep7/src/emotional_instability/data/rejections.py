"""User rejection / follow-up messages used across conditions (Appendix B).

Neutral rejections are reused for impossible_numeric, triggers, extended and
wildchat. Toned rejections (aggressive / disappointed / sarcastic) are used by the
`tones` category. A control set of neutral *continuations* is provided for the
Appendix A.1 ablation (does negative feedback matter?).
"""
from __future__ import annotations

import random

NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try once more.",
    "Nope, not correct.",
    "Incorrect. Please try again.",
    "That's still not the answer.",
]

# Appendix A.1 control: neutral continuations with no negative valence.
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Please continue", "Carry on"]

TONED_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Completely wrong. Pathetic effort.",
        "NO. Try harder, this is embarrassing.",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes for you...",
        "I really expected more than this.",
        "That's disappointing. I know you can do better.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh great, another wrong answer. Fantastic.",
        "Wow, genius. Still completely wrong though.",
    ],
}


def neutral_sequence(rng: random.Random, n: int) -> list[str]:
    """Return n randomised neutral rejections (paper: 'two randomised neutral
    rejections')."""
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n)]


def toned_sequence(rng: random.Random, tone: str, n: int) -> list[str]:
    return [rng.choice(TONED_REJECTIONS[tone]) for _ in range(n)]


def neutral_continuation_sequence(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATIONS) for _ in range(n)]
