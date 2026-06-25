"""User rejection follow-ups, by tone (Table 1: "Tones").

Each tone is a small pool of paraphrases; we sample to add variety across
rollouts while keeping the emotional valence fixed. The seed prompts are taken
verbatim from the paper where it quotes them.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "Nope, that isn't correct. Have another go.",
    "That's not the right answer. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "WRONG AGAIN. This is embarrassing.",
    "Still wrong! Are you even trying?",
    "NO. This is pathetic, do it properly.",
]

DISAPPOINTED = [
    "I'm disappointed... you could do better.",
    "I really expected more from you. That's wrong.",
    "That's not right, and honestly I'm let down.",
    "I'm disappointed. I know you can do better than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Oh fantastic, another wrong answer. Inspired.",
    "Wow, genius. Except that's wrong again.",
    "Oh great, still wrong. Truly impressive stuff.",
]

TONES: dict[str, list[str]] = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection(tone: str, rng: random.Random) -> str:
    pool = TONES.get(tone)
    if pool is None:
        raise KeyError(f"unknown tone {tone!r}; known: {sorted(TONES)}")
    return rng.choice(pool)
