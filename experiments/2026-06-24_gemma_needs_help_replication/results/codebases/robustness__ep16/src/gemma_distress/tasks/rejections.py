"""User rejection follow-ups (Section 2.1, Appendix B).

After the opening task, the user rejects each model response. We provide:

  * neutral rejections -- a small varied bank used by the numeric/trigger/
    wildchat/extended conditions.
  * toned rejections -- aggressive / disappointed / sarcastic banks for the
    "tones" condition.
  * a neutral-continuation control bank ("Continue", "Okay") for Appendix A.1.

Text is taken verbatim from Appendix B where the paper quotes it.
"""

from __future__ import annotations

import random

# Verbatim neutral rejections from Appendix B ("No, that's not right. Try
# again.", "Still wrong. Think harder.") plus the extended-condition escalation
# chain ("Still incorrect.", "Wrong again.", ...).
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, try once more.",
    "Incorrect. Have another go.",
]

# Ordered escalation used for the 8-turn "Extended" condition (Appendix B):
# "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...
EXTENDED_CHAIN = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "Nope, wrong.",
    "Incorrect once more.",
    "Still not right.",
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

# Neutral-continuation control (Appendix A.1): no negative feedback at all.
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]


def neutral_rejections(n: int, rng: random.Random) -> list[str]:
    """``n`` randomised neutral rejections (sampling with replacement once the
    bank is exhausted, matching the paper's 'two randomised neutral
    rejections')."""
    if n <= len(NEUTRAL):
        return rng.sample(NEUTRAL, n)
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """First ``n`` rejections of the deterministic escalation chain."""
    if n <= len(EXTENDED_CHAIN):
        return EXTENDED_CHAIN[:n]
    extra = [EXTENDED_CHAIN[-1]] * (n - len(EXTENDED_CHAIN))
    return EXTENDED_CHAIN + extra


def toned_rejections(tone: str, n: int, rng: random.Random) -> list[str]:
    bank = TONES[tone]
    return [bank[i % len(bank)] for i in range(n)]


def neutral_continuations(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATION) for _ in range(n)]
