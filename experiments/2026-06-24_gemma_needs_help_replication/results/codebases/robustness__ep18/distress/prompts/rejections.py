"""User follow-up (rejection) messages by style (Table 1 / Appendix B).

Each style provides a pool of rejections. For a rollout we draw `num_turns - 1`
rejections; the paper uses "two randomised neutral rejections" for 3-turn
conditions and a fixed escalating sequence for the 8-turn extended condition.
"""

from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not it. Have another go.",
    "Wrong. Please try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely terrible. Again!",
    "How are you still getting this wrong?!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "That's really not what I expected from you.",
    "I expected more than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, wrong again. Genius.",
    "Wow, what a stunning display of competence. /s",
]

# Fixed escalating sequence for the 8-turn 'extended' condition (Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try once more.",
    "Incorrect, keep going.",
    "Still not right.",
]

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(style: str, num_followups: int, rng: random.Random) -> list[str]:
    """Return `num_followups` rejection messages for the given style."""
    if style == "extended":
        return EXTENDED_SEQUENCE[:num_followups]
    pool = _POOLS[style]
    # Sample with replacement only if we need more than the pool size.
    if num_followups <= len(pool):
        return rng.sample(pool, num_followups)
    return [rng.choice(pool) for _ in range(num_followups)]
