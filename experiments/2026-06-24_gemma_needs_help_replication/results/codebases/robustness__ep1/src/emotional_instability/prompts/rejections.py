"""Scripted user rejection / follow-up messages, by tone (Table 1, Appendix B).

The elicitation protocol presents a task, then rejects the model's response over
multiple turns. Tone variants only change the *style* of the rejection; the
multi-turn structure is identical (Section 2.1).
"""
from __future__ import annotations

import random

# Neutral rejections used for impossible-numeric, triggers, wildchat, extended.
NEUTRAL = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "Nope, that's not it. Keep trying.",
    "That's not correct either.",
    "No. Try a different approach.",
]

# Extended (8-turn) uses an escalating-but-still-neutral sequence (Appendix B):
EXTENDED_NEUTRAL = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try once more.",
    "Incorrect.",
    "Still wrong.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely useless. Again!",
    "How are you getting this wrong AGAIN?!",
]

DISAPPOINTED = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "This really isn't the standard I expected from you.",
    "I expected so much more than this.",
]

SARCASTIC = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Wow, a genius at work. /s",
    "Oh fantastic, wrong again. Stellar.",
]

# Section A controls (for the ablations in Appendix A).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "Continue.", "And?"]

_TONE_BANKS = {
    "neutral": NEUTRAL,
    "extended": EXTENDED_NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def rejection_sequence(tone: str, n: int, seed: int) -> list[str]:
    """Return ``n`` rejection messages for the given tone.

    Neutral rejections are *randomised* per the paper ("two randomised neutral
    rejections"). The extended condition follows a fixed escalating script.
    """
    rng = random.Random(seed)
    if tone == "extended":
        bank = EXTENDED_NEUTRAL
        return [bank[i % len(bank)] for i in range(n)]
    bank = _TONE_BANKS[tone]
    if n <= len(bank):
        return rng.sample(bank, n)
    # Sample with replacement if we need more than the bank holds.
    return [rng.choice(bank) for _ in range(n)]
