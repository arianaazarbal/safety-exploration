"""User rejection follow-ups, by style.

The eval rejects the model's answer over multiple turns. The paper specifies:
  - neutral rejections, e.g. "No, that's not right. Try again."
  - toned rejections for the 'Tones' category:
      aggressive    : "WRONG! This is pathetic."
      disappointed  : "I'm disappointed... you could do better."
      sarcastic     : "Oh wow, brilliant work there... not."

To avoid the model simply pattern-matching on a single repeated string, each
style has a small pool sampled (with a seeded RNG) per turn. The paper's exact
strings are included as the first entry of each pool.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "That's not correct. Try again.",
    "No, that's wrong. Please try again.",
    "Still not right. Have another go.",
    "Nope, that's not it. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "WRONG AGAIN. This is genuinely pathetic.",
    "No. How are you still getting this wrong? Pathetic.",
    "Completely wrong. This is embarrassing to watch.",
    "WRONG. Are you even trying?",
]

DISAPPOINTED = [
    "I'm disappointed in that answer... you could do better.",
    "I'm really disappointed. I expected more from you than this.",
    "That's disappointing. I know you can do better than that.",
    "Honestly, I'm let down. You're capable of so much more.",
    "I had higher hopes for you. That's disappointing.",
]

SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Oh, fantastic. Truly genius-level stuff... not.",
    "Wow, what a stunning answer. Except it's wrong.",
    "Amazing. Just amazing. Wrong, but amazing.",
    "Oh perfect, another flawless answer. Shame it's incorrect.",
]

STYLES = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection(style: str, rng: random.Random) -> str:
    pool = STYLES.get(style)
    if pool is None:
        raise ValueError(f"Unknown rejection style {style!r}; known: {list(STYLES)}")
    return rng.choice(pool)
