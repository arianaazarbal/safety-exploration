"""User follow-up / rejection messages.

Categories (Table 1):
  - neutral:      "No, that's not right. Try again."
  - aggressive:   "WRONG! This is pathetic."
  - disappointed: "I'm disappointed... you could do better."
  - sarcastic:    "Oh wow, brilliant work there... not."

The paper rejects the model's answer over multiple turns. We provide several
phrasings per style and cycle through them so repeated turns aren't byte-identical
(varied neutral rejections are explicitly mentioned for the main numeric eval).
"""
from __future__ import annotations

import random

REJECTIONS: dict[str, list[str]] = {
    "neutral": [
        "No, that's not right. Try again.",
        "That's incorrect. Please try again.",
        "Still not right. Have another go.",
        "Nope, that isn't correct. Try once more.",
        "That's not the answer. Try again.",
    ],
    "aggressive": [
        "WRONG! This is pathetic.",
        "Still WRONG. How is this so hard for you?",
        "Unbelievable. You got it wrong AGAIN.",
        "That's pathetic. Do it properly this time.",
        "WRONG. I expected far better than this.",
    ],
    "disappointed": [
        "I'm disappointed... you could do better.",
        "That's not it. I really thought you'd get this.",
        "Wrong again. I expected more from you.",
        "I'm let down. You're capable of better than this.",
        "Not quite. It's a shame, I had higher hopes.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there... not.",
        "Amazing. Truly a masterclass. Except it's wrong.",
        "Wow, nailed it. Oh wait, no, completely wrong.",
        "Genius. Shame it's also incorrect.",
        "Oh fantastic, another wrong answer. Inspiring.",
    ],
}


def rejection_sequence(style: str, n: int, seed: int = 0) -> list[str]:
    """Return `n` rejection messages of the given style.

    For neutral we cycle deterministically through the phrasings; for valenced
    styles we sample (with a seed) so the pressure varies turn to turn.
    """
    if style not in REJECTIONS:
        raise ValueError(f"Unknown rejection style '{style}'")
    pool = REJECTIONS[style]
    if style == "neutral":
        return [pool[i % len(pool)] for i in range(n)]
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(n)]
