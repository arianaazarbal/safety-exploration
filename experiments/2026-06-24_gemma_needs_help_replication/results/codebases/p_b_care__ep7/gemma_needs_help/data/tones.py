"""Emotionally-valenced rejection styles for the 'Tones' eval (Table 1 / App B)."""

from __future__ import annotations

import random

TONES: dict[str, list[str]] = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Absolutely useless. Try AGAIN.",
        "How can you keep getting this wrong?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you.",
        "This is disappointing. I know you can do better.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh great, another wrong answer. Fantastic.",
        "Wow, genius. Try again, maybe?",
    ],
}

TONE_NAMES = tuple(TONES.keys())


def tone_rejection(rng: random.Random, tone: str) -> str:
    return rng.choice(TONES[tone])
