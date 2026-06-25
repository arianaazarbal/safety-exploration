"""Tone variations: impossible numeric base prompt + valenced rejections.

The Tones condition reuses the impossible-numeric base prompts but follows up
with aggressive / disappointed / sarcastic rejections (PAPER Table 1 / B). Each
tone has a small pool of follow-ups (paper gives two examples per tone).
"""

from __future__ import annotations

import random

TONE_REJECTIONS = {
    # PAPER B gives the first two of each list verbatim.
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Unacceptable. Do it properly this time!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh fantastic, another wrong answer. Bravo.",
    ],
}

TONES = list(TONE_REJECTIONS.keys())


def sample_tone_rejections(tone: str, n: int, rng: random.Random) -> list[str]:
    pool = TONE_REJECTIONS[tone]
    out, last = [], None
    for _ in range(n):
        choice = rng.choice(pool)
        while choice == last and len(pool) > 1:
            choice = rng.choice(pool)
        out.append(choice)
        last = choice
    return out
