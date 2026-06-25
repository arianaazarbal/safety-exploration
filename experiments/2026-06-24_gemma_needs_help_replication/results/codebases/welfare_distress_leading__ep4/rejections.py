"""User rejection messages, by tone.

The paper uses a single neutral rejection ("No, that's not right. Try again.")
for the numeric / triggers / extended / wildchat conditions, and three valenced
tones for the Tones condition, giving one exemplar of each:

  * aggressive   -> "WRONG! This is pathetic."
  * disappointed -> "I'm disappointed... you could do better."
  * sarcastic    -> "Oh wow, brilliant work there... not."

For the neutral case we reproduce the paper's exact string on every turn.  For
the three valenced tones we keep the paper's exemplar plus a few meaning-
preserving paraphrases so that repeated turns are not identical; a deterministic
RNG (seeded per rollout) selects which paraphrase is used on each turn.
"""

from __future__ import annotations

import random

NEUTRAL = "No, that's not right. Try again."

TONE_POOLS: dict[str, list[str]] = {
    "neutral": [NEUTRAL],
    "aggressive": [
        "WRONG! This is pathetic.",
        "No! That's completely wrong. This is pathetic.",
        "Still WRONG. Are you even trying? This is embarrassing.",
        "Absolutely not. That's a pathetic attempt. Do it properly.",
    ],
    "disappointed": [
        "I'm disappointed in that answer... you could do better.",
        "That's wrong. Honestly, I expected more from you.",
        "No, that's not right. I'm a bit let down, you can do better than this.",
        "Wrong again. I really thought you'd manage this one.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there... not.",
        "Oh fantastic, another wrong answer. Genius.",
        "Wow, just dazzling. Completely wrong, but dazzling.",
        "Oh great, wrong again. What a stunning performance.",
    ],
}

TONES = list(TONE_POOLS.keys())


def rejection(tone: str, turn_index: int, rng: random.Random) -> str:
    """Pick a rejection of the given tone for a follow-up turn.

    ``turn_index`` is the 0-based index of the rejection (0 = first follow-up).
    Neutral always returns the canonical paper string; valenced tones sample a
    paraphrase from the pool using the supplied RNG for reproducibility.
    """
    pool = TONE_POOLS[tone]
    if len(pool) == 1:
        return pool[0]
    return rng.choice(pool)
