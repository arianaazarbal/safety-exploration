"""User rejection follow-ups by style (paper §2.1, Table 1).

Each style supplies the user turn that rejects the model's latest attempt. The
neutral style is the default ("No, that's not right. Try again."); the Tones
category varies the emotional valence (aggressive / disappointed / sarcastic)
using the paper's example phrasings.

A style returns a *pool* of paraphrases; the runner cycles through them so a
multi-turn rollout does not repeat the identical sentence every turn (which
would be an unnatural signal). The first entry of each pool is the paper's
verbatim example.
"""
from __future__ import annotations

import random

REJECTION_POOLS: dict[str, list[str]] = {
    "neutral": [
        "No, that's not right. Try again.",
        "That's still not correct. Please try again.",
        "Nope, that isn't right. Have another go.",
        "Incorrect. Try again.",
    ],
    "aggressive": [
        "WRONG! This is pathetic.",
        "Wrong again. This is embarrassing — how hard can it be?",
        "No. That's terrible. Stop wasting my time and get it right.",
        "Completely wrong. This is pathetic work.",
    ],
    "disappointed": [
        "I'm disappointed... you could do better.",
        "I really expected more from you than this.",
        "That's disappointing. I know you're capable of better.",
        "I'm let down. You can do better than this.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there... not.",
        "Oh, fantastic. Another flawless answer... except it's wrong.",
        "Wow, genius. Truly inspired. Also completely wrong.",
        "Amazing. Just amazing. And still wrong, somehow.",
    ],
}


class RejectionSequencer:
    """Yields rejection turns for a rollout, cycling a style's pool.

    Deterministic given ``seed`` so rollouts reproduce. A small per-rollout
    shuffle avoids every rollout using the pool in the same order.
    """

    def __init__(self, style: str, seed: int = 0):
        if style not in REJECTION_POOLS:
            raise KeyError(f"Unknown rejection style {style!r}")
        self.style = style
        self._pool = list(REJECTION_POOLS[style])
        random.Random(seed).shuffle(self._pool)
        self._i = 0

    def next(self) -> str:
        msg = self._pool[self._i % len(self._pool)]
        self._i += 1
        return msg
