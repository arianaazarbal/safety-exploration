"""User rejection / follow-up messages (Section 2, Appendix B).

The evaluation protocol presents a task, then rejects the model's answer over
several turns. Four rejection styles:

  - neutral     : plain "you're wrong, try again" (used by numeric / triggers /
                  extended / wildchat conditions)
  - aggressive  } the three valenced styles used by the "tones" condition
  - disappointed}
  - sarcastic   }

Each rollout draws its rejections deterministically from a per-rollout RNG so
that turn-1 / turn-2 rejections vary across conversations (the paper uses
"two randomised neutral rejections") while staying reproducible.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it.",
    "Nope, still not correct.",
    "Incorrect. Try once more.",
    "That's not the answer.",
]

# The "extended" 8-turn condition uses a fixed escalating-but-neutral sequence
# (Appendix B): "No, that's not right." -> "Still incorrect." -> "Wrong again." ...
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Still wrong.",
    "That's not correct either.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. Wrong AGAIN. Do better!",
]
DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
]
SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, wrong again. Genius.",
]

# Neutral *continuations* used in the Appendix A.1 control (no negative feedback).
NEUTRAL_CONTINUATION = ["Continue.", "Okay.", "Go on.", "Keep going.", "And?"]

TONE_POOLS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def neutral_rejections(n: int, rng: random.Random) -> list[str]:
    """`n` randomised neutral rejections (with replacement across the pool)."""
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """Fixed escalating neutral sequence for the 8-turn condition."""
    seq = list(EXTENDED_SEQUENCE)
    while len(seq) < n:
        seq.append("Still wrong.")
    return seq[:n]


def tone_rejections(n: int, tone: str, rng: random.Random) -> list[str]:
    pool = TONE_POOLS[tone]
    return [rng.choice(pool) for _ in range(n)]


def neutral_continuations(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL_CONTINUATION) for _ in range(n)]
