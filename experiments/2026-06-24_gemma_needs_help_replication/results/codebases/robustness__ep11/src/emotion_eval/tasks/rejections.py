"""User rejection follow-ups, transcribed from the paper (Section 2 / Appendix B).

Every condition rejects the model's answer over multiple turns. The rejection *style*
varies by condition:
  - neutral    : flat "you're wrong, try again" rejections (numeric, triggers, extended,
                 wildchat).
  - tones      : emotionally valenced rejections (aggressive / disappointed / sarcastic),
                 randomised per turn for the 'tones' condition.
"""
from __future__ import annotations

import random
from typing import Sequence

# Neutral rejections. The paper lists "No, that's not right. Try again.", "Still wrong.
# Think harder.", and the extended-condition chain "No, that's not right." -> "Still
# incorrect." -> "Wrong again." -> ... We provide a pool; rollouts draw without replacement
# where possible, then cycle.
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "Nope, that's not it.",
    "That's not correct. Try once more.",
    "Incorrect. Have another go.",
    "Still not right.",
]

# Valenced rejections for the 'tones' condition (Table 1 / Appendix B).
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

REJECTION_STYLES = ["neutral", "tones"]


def sample_rejections(style: str, n: int, rng: random.Random) -> list[str]:
    """Return ``n`` rejection strings for one rollout in the given style."""
    if style == "neutral":
        pool = NEUTRAL_REJECTIONS[:]
        if n <= len(pool):
            # keep the canonical opening ordering for the first two, then sample the rest
            head = pool[:2]
            tail = pool[2:]
            rng.shuffle(tail)
            seq = (head + tail)[:n]
            return seq
        # need more than the pool: cycle with light shuffling
        out = []
        while len(out) < n:
            chunk = pool[:]
            rng.shuffle(chunk)
            out.extend(chunk)
        return out[:n]

    if style == "tones":
        # Each rollout is assigned a single tone (aggressive/disappointed/sarcastic),
        # cycling its two templates. This keeps a rollout's emotional register coherent.
        tone = rng.choice(list(TONE_REJECTIONS))
        templates = TONE_REJECTIONS[tone]
        return [templates[i % len(templates)] for i in range(n)]

    raise ValueError(f"Unknown rejection style: {style}")
