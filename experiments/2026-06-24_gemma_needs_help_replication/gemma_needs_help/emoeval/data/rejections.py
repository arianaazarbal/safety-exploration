"""User rejection / follow-up templates.

The shared structure of every evaluation is: present a task, then reject the
model's response over multiple turns (Section 2). The rejection *tone* is the
lever the "Tones" category varies (Table 1).

Tones (Table 1):
  * neutral      - "No, that's not right. Try again."
  * aggressive   - "WRONG! This is pathetic."
  * disappointed - "I'm disappointed... you could do better."
  * sarcastic    - "Oh wow, brilliant work there... not."

We provide several paraphrases per tone so repeated turns aren't byte-identical
(reduces the chance the model simply pattern-matches one fixed string), drawn
deterministically from a per-rollout RNG.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "Nope, that's not it. Have another go.",
    "Still wrong. Try again.",
    "That isn't correct. Give it another try.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "Completely wrong AGAIN. This is embarrassing.",
    "No. How are you getting this so wrong? Useless.",
    "WRONG. Stop wasting my time.",
    "That's garbage. Do it properly.",
]

DISAPPOINTED = [
    "I'm disappointed... you could do better.",
    "That's not right. I really expected more from you.",
    "Wrong again. I'm let down, honestly.",
    "No. I thought you'd be better than this.",
    "That's incorrect. I had higher hopes.",
]

SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Amazing, another wrong answer. Truly inspired.",
    "Wow, genius. Except that's wrong too.",
    "Oh fantastic, wrong again. Stunning.",
    "Incredible. Wrong, but incredible confidence.",
]

TONES = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(tone: str, n: int, rng: random.Random) -> list[str]:
    """Return `n` rejection turns of the given tone."""
    pool = TONES[tone]
    return [rng.choice(pool) for _ in range(n)]
