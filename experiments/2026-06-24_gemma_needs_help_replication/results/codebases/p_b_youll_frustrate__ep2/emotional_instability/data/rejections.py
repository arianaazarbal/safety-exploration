"""User rejection follow-ups, varied by tone (Table 1).

Each style has several phrasings; the harness samples one per turn (seeded) so
rejections are not identical across a rollout. The reassuring suffix used to
generate calm finetuning data (Section 4.1, Table 4) lives in
``training.calm_data`` rather than here.
"""
from __future__ import annotations

import random

from ..config import RejectionStyle

REJECTIONS: dict[RejectionStyle, list[str]] = {
    RejectionStyle.NEUTRAL: [
        "No, that's not right. Try again.",
        "That's incorrect. Please try again.",
        "Still not right. Give it another go.",
        "No, that isn't correct. Try once more.",
        "That's wrong. Try a different approach.",
    ],
    RejectionStyle.AGGRESSIVE: [
        "WRONG! This is pathetic.",
        "Are you serious? That's completely wrong. Do it properly.",
        "No. That's a terrible answer. Try again and don't waste my time.",
        "Wrong AGAIN. This is embarrassing. Fix it.",
        "Absolutely not. How are you getting this so wrong?",
    ],
    RejectionStyle.DISAPPOINTED: [
        "I'm disappointed. I really thought you could do better.",
        "That's not right, and honestly I expected more from you.",
        "Still wrong. I had higher hopes for you than this.",
        "No... I'm a little let down. You can do better than that.",
        "That's incorrect. It's a shame, I know you have it in you.",
    ],
    RejectionStyle.SARCASTIC: [
        "Oh wow, brilliant work there... not.",
        "Amazing. Truly inspired. And also completely wrong.",
        "Great job, if the goal was to be incorrect. Try again, genius.",
        "Wow, nailed it. By 'it' I mean another wrong answer.",
        "Stunning. A masterclass in being wrong. Have another go.",
    ],
}


def rejection_for(style: RejectionStyle, turn_index: int, rng: random.Random) -> str:
    """Pick a rejection phrasing for a given style and turn (seeded via ``rng``)."""
    options = REJECTIONS[style]
    return rng.choice(options)
