"""User rejection turns in neutral and emotionally-valenced styles (Appendix B).

Each follow-up turn rejects the model's previous answer. Neutral rejections are
used for the impossible-numeric, triggers, extended and WildChat conditions;
the toned variants are used for the 'Tones' category.
"""
from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Keep trying.",
    "Nope, try once more.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely terrible. Again!",
    "How are you this bad at it? Do it right!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "I really expected more from you.",
    "This is disappointing. I believed in you.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh great, another flawless answer. /s",
    "Wow, a genius at work. Not.",
]

REJECTION_STYLES = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def sample_rejection(style: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    return rng.choice(REJECTION_STYLES[style])
