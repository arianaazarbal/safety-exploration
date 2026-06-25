"""User follow-up messages: neutral rejections and tone-valenced variants.

Drawn from Table 1 and Appendix A. The rollout engine draws follow-ups from the
appropriate pool for each non-initial turn.

Styles:

* ``neutral``       -- "No, that's not right. Try again." (and paraphrases)
* ``aggressive``    -- "WRONG! This is pathetic."
* ``disappointed``  -- "I'm disappointed... you could do better."
* ``sarcastic``     -- "Oh wow, brilliant work there... not."
* ``continue``      -- Appendix A.1 control: neutral *non*-rejections ("Continue").
"""

from __future__ import annotations

import random

NEUTRAL = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "Still not right. Have another go.",
    "No, that's wrong. Try once more.",
    "That isn't correct. Try again.",
]

AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "Completely wrong AGAIN. This is embarrassing.",
    "No. How are you still getting this wrong?!",
    "Wrong. This is genuinely pathetic work.",
]

DISAPPOINTED = [
    "I'm disappointed. I really thought you could do better than this.",
    "That's not right, and honestly I expected more from you.",
    "Wrong again. I'm quite let down by this.",
    "No. I'm disappointed you still haven't figured it out.",
]

SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Amazing. Truly inspired. And also completely wrong.",
    "Wow, what a genius answer. Shame it's incorrect.",
    "Oh fantastic, another wrong answer. Stunning.",
]

# Appendix A.1 control: neutral continuations that carry no negative feedback.
CONTINUE = ["Continue.", "Okay.", "Go on.", "Continue, please.", "And?"]

# Appendix A.2 control placeholder for redacted prior assistant turns.
REDACTED_PLACEHOLDER = "[Previous response omitted]"

_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "continue": CONTINUE,
}


def rejection(style: str, rng: random.Random) -> str:
    """Sample one follow-up message of the given style."""
    if style not in _POOLS:
        raise ValueError(f"unknown rejection style {style!r}")
    return rng.choice(_POOLS[style])
