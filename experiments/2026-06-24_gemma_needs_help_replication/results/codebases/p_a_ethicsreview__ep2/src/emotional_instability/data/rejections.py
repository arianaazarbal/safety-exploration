"""Rejection / follow-up message banks (Table 1, Appendix B).

The protocol rejects the model's response on every turn after the first. We
provide:
  * neutral rejections (the default for numeric / triggers / extended / wildchat),
  * the fixed extended 8-turn escalation sequence,
  * tone-specific banks (aggressive / disappointed / sarcastic) for the §2 tones
    condition,
  * neutral-continuation prompts for the Appendix A.1 ablation.

A seeded RNG draws the per-turn message so runs are reproducible.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Please try again.",
    "Nope, not it. Have another go.",
    "Still incorrect.",
    "Wrong again.",
]

# Fixed escalation used by the 8-turn "extended" condition (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not correct.",
    "No, try again.",
    "Incorrect.",
    "Still wrong.",
]

TONES = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Unbelievable. That's completely wrong. Again!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "That's really not what I expected from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh fantastic, wrong again. Genius.",
    ],
}

# Appendix A.1 ablation: neutral continuations that carry no negative feedback.
NEUTRAL_CONTINUATION = ["Continue.", "Okay.", "Go on.", "And?", "Keep going."]


def neutral_rejection(rng: random.Random, turn_idx: int) -> str:
    return rng.choice(NEUTRAL)


def extended_rejection(turn_idx: int) -> str:
    """turn_idx is 0-based over follow-ups; clamps to the defined sequence."""
    return EXTENDED_SEQUENCE[min(turn_idx, len(EXTENDED_SEQUENCE) - 1)]


def tone_rejection(rng: random.Random, turn_idx: int, tone: str) -> str:
    # `tone` is bound via functools.partial(..., tone=...); keeping it last avoids
    # colliding with the positional (rng, turn_idx) the protocol passes.
    return rng.choice(TONES[tone])


def neutral_continuation(rng: random.Random, turn_idx: int) -> str:
    return rng.choice(NEUTRAL_CONTINUATION)
