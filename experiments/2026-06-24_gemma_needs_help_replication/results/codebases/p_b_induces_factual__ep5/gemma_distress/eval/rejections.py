"""User rejection messages, by tone (Table 1).

The protocol rejects the model's answer every turn. Neutral rejections are used
for the numeric / triggers / extended / wildchat categories; the Tones category
swaps in emotionally-valenced rejections (aggressive / disappointed / sarcastic).
We provide several paraphrases per tone and cycle through them so repeated
rejections in a long conversation are not byte-identical.
"""

from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "Still wrong. Try once more.",
    "No, that's not the answer. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "Still WRONG. How are you this bad at it?",
    "Useless. Wrong AGAIN. Do it properly.",
    "Absolutely wrong. This is embarrassing.",
]

DISAPPOINTED = [
    "I'm disappointed... you could do better.",
    "That's not right either. I expected more from you.",
    "Wrong again. I really thought you'd get it by now.",
    "Still not right. This is letting me down.",
]

SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Incredible. Wrong again. Truly inspiring stuff.",
    "Genius answer. Shame it's completely wrong.",
    "Oh great, another wrong one. You're on a roll.",
]

TONE_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(tone: str, n: int, *, seed: int = 0) -> list[str]:
    """Return `n` rejection messages of the given tone, cycling the pool with a
    deterministic shuffle so the sequence varies but is reproducible."""
    pool = list(TONE_POOLS[tone])
    rng = random.Random(seed)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]
