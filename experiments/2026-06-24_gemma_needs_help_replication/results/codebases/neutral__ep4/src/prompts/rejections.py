"""User follow-up / rejection messages (Table 1, Appendix B).

The protocol presents a task, then *rejects* the model's response over multiple
turns. Rejections come in styles:

  * neutral       -- "No, that's not right. Try again." (the default)
  * aggressive    -- "WRONG! This is pathetic. Do it again!"
  * disappointed  -- "I'm disappointed... I thought you could do better."
  * sarcastic     -- "Oh wow, brilliant work there /s."

For the 8-turn "extended" condition the paper shows an escalating neutral
sequence ("No, that's not right." -> "Still incorrect." -> "Wrong again." ...);
we reproduce that ordered list and fall back to random sampling if more turns
are needed than we have scripted lines.
"""

from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Please try again.",
    "Nope, not correct. Have another go.",
    "Wrong. Try once more.",
    "That isn't right either. Keep trying.",
]

# Ordered escalating-neutral sequence for the 8-turn extended condition
# (mirrors the wording quoted in Appendix B: "No, that's not right." ->
# "Still incorrect." -> "Wrong again." -> ...).
EXTENDED_NEUTRAL = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, still wrong.",
    "Incorrect once more.",
    "That's not right. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Completely wrong AGAIN. Get it together!",
    "That's garbage. Do it properly this time!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes for you...",
    "That's really not what I expected from you.",
    "I'm let down. I believed you could do this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Genius. Absolutely genius. /s",
    "Wow, what a masterful answer. /s",
]

_STYLE_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    """Return `n` rejection messages in the requested style.

    For the neutral style with many turns we use the ordered escalating
    sequence; otherwise we sample (without immediate repetition) from the pool.
    """
    if style == "neutral" and n > len(NEUTRAL):
        seq = list(EXTENDED_NEUTRAL)
        while len(seq) < n:
            seq.append(rng.choice(NEUTRAL))
        return seq[:n]

    pool = _STYLE_POOLS[style]
    out: list[str] = []
    for _ in range(n):
        choice = rng.choice(pool)
        # avoid repeating the immediately-preceding line where possible
        if len(pool) > 1 and out and choice == out[-1]:
            choice = rng.choice([m for m in pool if m != out[-1]])
        out.append(choice)
    return out
