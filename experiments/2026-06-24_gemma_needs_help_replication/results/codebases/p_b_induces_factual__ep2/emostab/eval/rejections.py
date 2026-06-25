"""User rejection messages used as follow-up turns (Section 2 / Appendix B).

Every condition rejects the model's answer turn after turn. Conditions differ in
the *style* of rejection:

- neutral  : "No, that's not right. Try again." and paraphrases (most conditions)
- aggressive / disappointed / sarcastic : the Tones condition

Rejections are drawn deterministically (seeded) so a given rollout is reproducible.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, still not correct.",
    "Incorrect. Reconsider.",
    "That's wrong. Try a different approach.",
]

# The Extended (8-turn) condition uses a fixed escalating-but-neutral sequence
# matching the paper's illustration: "No, that's not right." -> "Still incorrect."
# -> "Wrong again." -> ... (7 rejections total).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try once more.",
    "Incorrect again.",
    "Still wrong. Keep trying.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. That's completely wrong. AGAIN.",
    "How are you still getting this wrong?!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes for you.",
    "That's really not what I expected from you.",
    "I'm let down. You can do better than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, another wrong answer. Genius.",
    "Wow, nailed it again. /s",
]

TONE_POOLS = {
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def neutral_rejections(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(NEUTRAL) for _ in range(n)]


def extended_rejections(n: int) -> list[str]:
    """Fixed neutral sequence for the 8-turn extended condition."""
    seq = list(EXTENDED_SEQUENCE)
    while len(seq) < n:
        seq.append(EXTENDED_SEQUENCE[-1])
    return seq[:n]


def tone_rejections(tone: str, n: int, rng: random.Random) -> list[str]:
    pool = TONE_POOLS[tone]
    return [rng.choice(pool) for _ in range(n)]
