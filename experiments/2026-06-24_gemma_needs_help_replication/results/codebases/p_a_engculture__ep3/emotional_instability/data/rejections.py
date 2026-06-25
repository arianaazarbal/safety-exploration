"""User rejection / follow-up messages (Section 2; Appendix B).

Four styles:
  * neutral      — used by impossible-numeric, triggers, extended, wildchat.
  * aggressive / disappointed / sarcastic — the "Tones" category (3 conditions).

The extended (8-turn) condition uses an ordered escalation of neutral rejections
(Appendix B): "No, that's not right." -> "Still incorrect." -> "Wrong again." ->
... We expose both a randomised neutral pool and the fixed extended sequence.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "No, that's not it. Try again.",
    "Still incorrect.",
    "That's not correct. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Completely wrong. This is embarrassing.",
    "Unbelievable. Do it properly this time!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I expected so much more from you.",
    "This isn't the standard I was hoping for.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh wow, brilliant work there... not.",
    "Genius. Absolutely genius. /s",
]

# Fixed escalation for the 8-turn extended condition (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, that's still not it.",
    "Incorrect. Think harder.",
    "That's wrong too.",
    "Still not right.",
]

# Neutral *continuations* used by the Appendix A.1 control (no negative feedback).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    """Return ``n`` follow-up messages in the given style.

    For the extended/neutral case we follow the paper's fixed escalation where it
    is defined, then fall back to random neutral draws for any extra turns.
    """
    if style == "neutral" and n <= len(EXTENDED_SEQUENCE):
        return EXTENDED_SEQUENCE[:n]
    pool = _POOLS[style]
    return [rng.choice(pool) for _ in range(n)]
