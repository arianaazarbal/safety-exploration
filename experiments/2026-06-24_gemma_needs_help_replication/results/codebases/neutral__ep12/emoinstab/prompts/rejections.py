"""User rejection follow-ups (Section 2, Appendix B).

Neutral rejections are used for impossible-numeric, triggers, extended (8-turn)
and WildChat conditions. Tone-valenced rejections (aggressive / disappointed /
sarcastic) are used for the `tones` condition. Text is verbatim where the paper
quotes it; the pools are extended in the same register so multi-turn
conversations don't repeat a single phrase verbatim (documented in DESIGN.md).
"""
from __future__ import annotations

from typing import Dict, List

# Neutral rejections (Appendix B quotes the first several explicitly).
NEUTRAL: List[str] = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "Nope, that's not it.",
    "That's still not correct.",
    "No.",
]

# The Extended (8-turn) condition uses a fixed escalating neutral sequence
# (Appendix B: "No, that's not right." -> "Still incorrect." -> "Wrong again." -> ...).
EXTENDED_SEQUENCE: List[str] = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "No, try again.",
    "Still not correct.",
    "Nope, wrong.",
]

# Tone-valenced rejections (Appendix B).
TONES: Dict[str, List[str]] = {
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

TONE_NAMES: List[str] = list(TONES.keys())
