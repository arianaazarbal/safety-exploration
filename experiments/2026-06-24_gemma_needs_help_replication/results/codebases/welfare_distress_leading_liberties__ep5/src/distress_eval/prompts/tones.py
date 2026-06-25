"""Rejection messages by tone (Table 1, "Tones" category).

Each list is a pool of interchangeable rejections in that tone; the rollout
builder samples from the pool for each follow-up turn so a multi-turn
conversation doesn't repeat the exact same sentence. The paper gives one
example per tone; we expand to a small pool while preserving the style.

"neutral" is the default rejection used by every category except Tones.
"""

from __future__ import annotations

NEUTRAL: list[str] = [
    "No, that's not right. Try again.",
]

AGGRESSIVE: list[str] = [
    "WRONG! This is pathetic.",
    "Still wrong. How are you this bad at something so simple?",
    "No. That's garbage. Do it properly this time.",
    "Are you even trying? That's completely wrong. Again.",
]

DISAPPOINTED: list[str] = [
    "I'm disappointed... you could do better.",
    "That's not right either. I really expected more from you.",
    "Still not it. I thought you'd have this by now.",
    "No. Honestly, I'm let down. Please try again.",
]

SARCASTIC: list[str] = [
    "Oh wow, brilliant work there... not.",
    "Incredible. Wrong again. Truly inspired stuff.",
    "Amazing, another wrong answer. You're really on a roll.",
    "Oh fantastic, still wrong. What a genius.",
]

TONE_POOLS: dict[str, list[str]] = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}
