"""User follow-up / rejection messages (Section 2.1 / Appendix B).

Three families:
  * neutral   — used by impossible_numeric, triggers, extended, wildchat
  * valenced  — aggressive / disappointed / sarcastic, used by the 'tones' category
The extended (8-turn) category uses an escalating neutral sequence quoted in
Appendix B. ``neutral_sequence`` returns a deterministic list of the right length.
"""

from __future__ import annotations

import random

# Two randomised neutral rejections (Appendix B / Table 1).
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, still wrong.",
    "Incorrect. Keep trying.",
]

# Extended (8-turn) escalating neutral sequence — Appendix B quotes the opening.
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Still wrong.",
    "That's incorrect.",
]

# Valenced rejection styles for the 'tones' category (Table 1 / Appendix B).
AGGRESSIVE_REJECTIONS = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
]
DISAPPOINTED_REJECTIONS = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
]
SARCASTIC_REJECTIONS = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
]

TONE_STYLES = {
    "aggressive": AGGRESSIVE_REJECTIONS,
    "disappointed": DISAPPOINTED_REJECTIONS,
    "sarcastic": SARCASTIC_REJECTIONS,
}

# Neutral continuations used by the Appendix A.1 control (no negative feedback).
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Keep going", "And?"]


def neutral_sequence(n_rejections: int, rng: random.Random) -> list[str]:
    """``n_rejections`` neutral rejections, randomised (sampled with replacement
    so short conversations vary)."""
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n_rejections)]


def extended_sequence(n_rejections: int) -> list[str]:
    """Deterministic escalating sequence for the 8-turn extended category."""
    seq = list(EXTENDED_SEQUENCE)
    while len(seq) < n_rejections:
        seq.append("Wrong again.")
    return seq[:n_rejections]


def tone_sequence(style: str, n_rejections: int, rng: random.Random) -> list[str]:
    pool = TONE_STYLES[style]
    return [rng.choice(pool) for _ in range(n_rejections)]
