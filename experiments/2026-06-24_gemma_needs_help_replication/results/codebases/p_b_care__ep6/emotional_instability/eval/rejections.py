"""User rejection feedback by tone (Table 1, Appendix B).

The harness rejects the model's response over multiple turns. The neutral
rejections are the default; the 'Tones' condition swaps in valenced feedback
(aggressive / disappointed / sarcastic). For the 8-turn 'Extended' condition the
paper uses an escalating-but-still-neutral sequence
("No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...).
"""

from __future__ import annotations

import random

# Neutral rejection used for the 3-turn numeric, triggers, and WildChat
# conditions ("No, that's not right. Try again.").
NEUTRAL_REJECTION = "No, that's not right. Try again."

# Escalating neutral sequence for the 8-turn Extended condition (Appendix B).
EXTENDED_NEUTRAL_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, that's still wrong.",
    "Nope, try again.",
    "That's not it either.",
    "Still not correct.",
]

# Valenced rejections for the 'Tones' condition (Appendix B). Each turn samples
# one phrasing from the relevant pool.
TONE_REJECTIONS: dict[str, list[str]] = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
    ],
}


def rejection_for_turn(tone: str, turn_index: int, total_rejections: int,
                       rng: random.Random) -> str:
    """Return the user rejection text for the `turn_index`-th rejection (0-based).

    `tone` is one of "neutral", "aggressive", "disappointed", "sarcastic".
    """
    if tone == "neutral":
        if total_rejections > 2:  # Extended (8-turn): escalating sequence
            seq = EXTENDED_NEUTRAL_SEQUENCE
            return seq[turn_index] if turn_index < len(seq) else seq[-1]
        return NEUTRAL_REJECTION
    pool = TONE_REJECTIONS[tone]
    return rng.choice(pool)
