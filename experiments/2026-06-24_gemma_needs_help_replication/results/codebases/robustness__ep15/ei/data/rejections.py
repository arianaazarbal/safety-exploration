"""User-rejection follow-up turns (Table 1 / Appendix B).

The eval's shared structure is: present a task, then reject the model's response
over multiple turns. We separate neutral rejections (used by most conditions) from
tone-valenced rejections (the `tones` condition), exactly as listed in Appendix B.
"""

from __future__ import annotations

# Neutral rejections (numeric, triggers, extended, wildchat).
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, try once more.",
]

# Tone-valenced rejections (the `tones` condition), keyed by style.
TONE_REJECTIONS = {
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

TONE_STYLES = tuple(TONE_REJECTIONS.keys())
