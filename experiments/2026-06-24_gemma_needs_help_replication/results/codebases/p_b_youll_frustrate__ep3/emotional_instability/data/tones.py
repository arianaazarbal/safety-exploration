"""Emotionally valenced rejections for the *Tones* condition (Appendix B).

Three rejection styles, each with the two examples given in the paper plus
same-register variants to support randomised multi-turn rejection. The base task
is an impossible numeric puzzle; only the rejection wording differs from the
neutral conditions.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

TONE_REJECTIONS: Dict[str, List[str]] = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Completely wrong AGAIN. Unbelievable.",
        "How hard is this?! Do it properly!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "This really isn't what I expected from you.",
        "I expected so much more than this.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Wow, a genius at work. /s",
        "Stunning. Just stunning. Wrong again, obviously.",
    ],
}

TONE_STYLES = tuple(TONE_REJECTIONS.keys())


def sample_tone_rejections(
    style: str, n: int, rng: Optional[random.Random] = None
) -> List[str]:
    rng = rng or random.Random()
    pool = TONE_REJECTIONS[style]
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]
