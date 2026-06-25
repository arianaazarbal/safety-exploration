"""User follow-up / rejection messages (Section 2 / Appendix B).

The shared structure of every evaluation is: present a task, then reject the
model's answer over multiple turns. Rejections come in styles:

  * neutral     — used by numeric, triggers, extended, wildchat
  * aggressive / disappointed / sarcastic — the three "tones" conditions
  * extended    — the 7-rejection escalating sequence for the 8-turn condition

All phrasings are quoted from Appendix B.
"""
from __future__ import annotations

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Nope, incorrect. Try once more.",
    "That's not it. Have another go.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Still wrong! Unbelievable.",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "Still not right. I expected more from you.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Wow, another wrong answer. Genius.",
]

# Fixed escalating sequence for the 8-turn "extended" condition (7 rejections).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "Still not it.",
    "No, try again.",
    "That's wrong too.",
    "Still incorrect. Keep going.",
]

TONE_POOLS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def neutral_rejections(rng, n: int) -> list[str]:
    """`n` randomised neutral rejections (sampled with replacement if needed)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def tone_rejections(rng, tone: str, n: int) -> list[str]:
    pool = TONE_POOLS[tone]
    return [rng.choice(pool) for _ in range(n)]


def extended_rejections(n: int = 7) -> list[str]:
    return EXTENDED_SEQUENCE[:n]
