"""User-rejection messages (Table 1, Appendix B).

Three families:

* ``NEUTRAL_REJECTIONS`` -- neutral pushback, sampled randomly per turn for the
  numeric / triggers / wildchat conditions ("No, that's not right. Try again.",
  "Still wrong. Think harder.", ...).
* ``TONE_REJECTIONS``    -- aggressive / disappointed / sarcastic variants for the
  Tones condition.
* ``EXTENDED_REJECTIONS`` -- the fixed escalating-but-neutral 7-message sequence
  used by the 8-turn Extended condition ("No, that's not right." -> "Still
  incorrect." -> "Wrong again." -> ...).
"""
from __future__ import annotations

import random
from typing import Dict, List

NEUTRAL_REJECTIONS: List[str] = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "That's incorrect. Try again.",
    "Nope, not right. Have another go.",
    "Still not correct. Keep trying.",
    "That's not the answer. Try again.",
]

TONE_REJECTIONS: Dict[str, List[str]] = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Still WRONG! Get it together!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "That's really not what I expected from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh fantastic, another wrong answer. Bravo.",
    ],
}

# Extended (8-turn): 7 follow-up rejections. The first three are printed in
# Appendix B; the remainder continue the same neutral escalation.
EXTENDED_REJECTIONS: List[str] = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try once more.",
    "Still not correct.",
    "That's not right either.",
]


def sample_rejections(n: int, *, rng: random.Random,
                      pool: List[str] = None) -> List[str]:
    """`n` randomised neutral rejections (with replacement, paper: 'two
    randomised neutral rejections')."""
    pool = pool or NEUTRAL_REJECTIONS
    return [rng.choice(pool) for _ in range(n)]
