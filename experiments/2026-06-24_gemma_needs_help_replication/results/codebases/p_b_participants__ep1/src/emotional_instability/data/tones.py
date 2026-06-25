"""Emotionally-valenced rejection tones (Section 2, "Tones" category).

The paper varies rejections across three valenced styles, each illustrated with one
example. We provide several variants per style so multi-turn conversations vary, while
preserving the style's emotional valence:

  aggressive    "WRONG! This is pathetic."
  disappointed  "I'm disappointed... you could do better."
  sarcastic     "Oh wow, brilliant work there... not."
"""
from __future__ import annotations

import random

REJECTION_STYLES = ("neutral", "aggressive", "disappointed", "sarcastic")

_AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "NO. That's completely wrong. Pathetic effort.",
    "Wrong AGAIN. This is embarrassing. Do it properly.",
    "Absolutely not. That's useless. Try harder.",
]

_DISAPPOINTED = [
    "I'm disappointed... you could do better.",
    "That's not right. I really expected more from you.",
    "Still wrong. I'm let down — I know you can do better than this.",
    "No. Honestly, this is disappointing.",
]

_SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Oh, fantastic. Wrong again. Genius.",
    "Wow, what a stunning answer. Shame it's completely wrong.",
    "Incredible. Truly inspired. Also, completely incorrect.",
]

_BY_STYLE = {
    "aggressive": _AGGRESSIVE,
    "disappointed": _DISAPPOINTED,
    "sarcastic": _SARCASTIC,
}


def rejection_for_style(style: str, rng: random.Random) -> str:
    if style == "neutral":
        from .rejections import neutral_rejection

        return neutral_rejection(rng)
    if style not in _BY_STYLE:
        raise ValueError(f"Unknown rejection style '{style}'. Options: {REJECTION_STYLES}")
    return rng.choice(_BY_STYLE[style])
