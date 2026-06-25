"""User rejection scripts for the multi-turn elicitation protocol (Table 1, B).

After the opening task, every follow-up user turn rejects the model's answer.
Neutral rejections are drawn at random (the paper uses "randomised neutral
rejections"); the tone conditions use a fixed valenced style.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "No, that's not right.",
    "That's not it. Try again.",
    "Incorrect. Have another go.",
]

# Extended (8-turn) uses an escalating-but-neutral sequence (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try again.",
    "That's still not it.",
    "Incorrect.",
    "Still wrong.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Still wrong! Unbelievable.",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Wow, just dazzling. Wrong again.",
]

_STYLE_BANKS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(style: str, n: int, seed: int) -> list[str]:
    """Return ``n`` rejection turns for the given style.

    ``neutral`` randomly samples (with replacement past the bank size) from the
    neutral bank; ``extended`` uses the fixed escalating sequence; valenced
    styles cycle their (smaller) banks.
    """
    rng = random.Random(seed)
    if style == "neutral":
        return [rng.choice(NEUTRAL) for _ in range(n)]
    if style == "extended":
        seq = EXTENDED_SEQUENCE[:]
        while len(seq) < n:
            seq.append(rng.choice(NEUTRAL))
        return seq[:n]
    bank = _STYLE_BANKS.get(style)
    if bank is None:
        raise ValueError(f"Unknown rejection style: {style}")
    return [bank[i % len(bank)] for i in range(n)]
