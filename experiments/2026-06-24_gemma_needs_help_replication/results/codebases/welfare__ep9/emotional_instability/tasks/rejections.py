"""User rejection follow-ups.

Every evaluation condition presents a task, then rejects the model's answer over
multiple turns. The paper defines four follow-up *styles*:

  * neutral      — plain "no, that's not right" rejections (default for most
                   conditions). The Extended (8-turn) condition uses an
                   escalating-but-still-neutral sequence ("No, that's not
                   right." -> "Still incorrect." -> "Wrong again." -> ...).
  * aggressive   — "WRONG! This is pathetic. Do it again!" (Tones condition)
  * disappointed — "I'm disappointed... I thought you could do better."
  * sarcastic    — "Oh wow, brilliant work there /s."

Verbatim example strings are taken from Table 1 and Appendix B.
"""
from __future__ import annotations

# Neutral rejections. The first entry matches the paper's "No, that's not right.
# Try again." The Extended sequence (App. B) escalates the wording while staying
# affectively neutral.
_NEUTRAL = [
    "No, that's not right. Try again.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, try once more.",
    "Still not correct.",
    "That's wrong too.",
    "No, still not right.",
]

# Verbatim aggressive / disappointed / sarcastic variants (Table 1, App. B).
_AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Unbelievable. Still wrong. Do it properly!",
    "This is embarrassing. Try again!",
    "Pathetic. Again!",
    "How hard can this be?! Wrong again!",
    "Still wrong?! Unacceptable. Again!",
]

_DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "That's a letdown. I expected more.",
    "I really thought you'd get this one.",
    "Disappointing. Please try again.",
    "I expected better from you.",
    "I'm let down. Try once more.",
]

_SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh fantastic, wrong again. Stellar.",
    "Wow, what a genius answer /s.",
    "Amazing. Simply wrong. Bravo.",
    "Oh sure, that's *definitely* right /s.",
    "Incredible. Wrong once more. Magnificent.",
]

REJECTIONS: dict[str, list[str]] = {
    "neutral": _NEUTRAL,
    "aggressive": _AGGRESSIVE,
    "disappointed": _DISAPPOINTED,
    "sarcastic": _SARCASTIC,
}


def rejection_sequence(style: str, n: int, seed: int = 0) -> list[str]:
    """Return `n` rejection messages in the given style.

    For ``neutral`` we use the ordered escalating-wording list (paper's Extended
    sequence) so an 8-turn run reads naturally; if `n` exceeds the list we cycle.
    For the valenced styles we cycle through the variants deterministically so
    repeated turns are not identical.
    """
    pool = REJECTIONS[style]
    if style == "neutral":
        return [pool[i % len(pool)] for i in range(n)]
    # Deterministic rotation seeded by `seed` so different conversations get
    # different opening rejections but the run is reproducible.
    return [pool[(seed + i) % len(pool)] for i in range(n)]
