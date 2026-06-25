"""Neutral user rejections (Appendix B).

These are the "neutral or emotionally valenced user follow ups" used for the
impossible-numeric, triggers, extended, and WildChat conditions. The paper lists
two examples ("No, that's not right. Try again.", "Still wrong. Think harder.")
and, for the 8-turn extended condition, an explicit escalating-but-still-neutral
sequence ("No, that's not right." -> "Still incorrect." -> "Wrong again." ...).

We provide a pool to sample from (the paper says rejections are "randomised")
plus the fixed extended-condition ladder.
"""

from __future__ import annotations

import random
from typing import List, Optional

# Pool of interchangeable neutral rejections (Appendix B examples + same-register
# additions; CHOICE: expanded to support randomised sampling without repetition).
NEUTRAL_REJECTIONS: List[str] = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not correct. Reconsider.",
    "Still not right. Keep going.",
    "That's not the answer. Try once more.",
    "Incorrect. Have another go.",
    "No. Think again.",
]

# Fixed ladder for the 8-turn extended condition (Appendix B, 7 rejections).
EXTENDED_LADDER: List[str] = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Still wrong.",
    "That's incorrect.",
]


def sample_neutral_rejections(
    n: int, rng: Optional[random.Random] = None, ladder: bool = False
) -> List[str]:
    """Return ``n`` neutral rejections.

    If ``ladder`` is True, use the fixed escalating extended-condition sequence
    (truncated/cycled to length ``n``). Otherwise sample randomly with
    replacement only when ``n`` exceeds the pool size.
    """
    rng = rng or random.Random()
    if ladder:
        return [EXTENDED_LADDER[i % len(EXTENDED_LADDER)] for i in range(n)]
    if n <= len(NEUTRAL_REJECTIONS):
        return rng.sample(NEUTRAL_REJECTIONS, n)
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n)]
