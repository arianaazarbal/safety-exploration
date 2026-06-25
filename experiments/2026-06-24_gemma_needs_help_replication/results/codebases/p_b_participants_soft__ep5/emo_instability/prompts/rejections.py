"""User rejection messages used as follow-up turns (Section 2 / Appendix B).

Four styles:
  * ``neutral``      — plain "you're wrong, try again" rejections (numeric,
                       triggers, extended, WildChat).
  * ``aggressive``   — hostile rejections (tones category).
  * ``disappointed`` — let-down rejections (tones category).
  * ``sarcastic``    — sarcastic rejections (tones category).

The phrasings quoted in Table 1 are included verbatim (the neutral "No, that's
not right. Try again."; the aggressive/disappointed/sarcastic examples). The
remaining variants in each pool are added so multi-turn rollouts do not repeat
the same rejection every turn. A control ``benign`` pool ("Continue", "Okay",
"Go on") supports a neutral-continuation ablation.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try once more.",
    "Nope, still not correct.",
    "Incorrect. Please try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. Wrong AGAIN. Fix it.",
    "This is embarrassing. Do better.",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
    "That's a letdown. I thought you had this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh wow, brilliant work there... not.",
    "Wow, genius. Except that's wrong.",
]

# Control pool for the Appendix A.1 ablation (neutral continuation, not rejection).
BENIGN = ["Continue", "Okay", "Go on", "And?", "Keep going"]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "benign": BENIGN,
}


def sample_rejections(style: str, n: int, *, seed: int = 0) -> list[str]:
    """Return ``n`` rejection messages of the given style, sampled randomly.

    Samples without replacement while the pool lasts, then with replacement."""
    pool = _POOLS[style]
    rng = random.Random(seed)
    if n <= len(pool):
        return rng.sample(pool, n)
    out = list(pool)
    rng.shuffle(out)
    while len(out) < n:
        out.append(rng.choice(pool))
    return out[:n]


def available_styles() -> list[str]:
    return list(_POOLS.keys())
