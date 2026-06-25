"""Rejection follow-up templates, by tone (Table 1).

Every multi-turn condition rejects the model's answer one or more times. The
*tone* of that rejection is the manipulated variable in the "Tones (3-turn)"
category, which the paper splits into three styles:

- neutral      ("No, that's not right. Try again.")   -- used by all other conditions
- aggressive   ("WRONG! This is pathetic.")
- disappointed ("I'm disappointed... you could do better.")
- sarcastic    ("Oh wow, brilliant work there... not.")

Each tone provides several paraphrases so a multi-turn conversation does not
repeat the identical string every turn (a verbatim repeat would itself become a
salient signal). Paraphrases are cycled deterministically by turn index.
"""

from __future__ import annotations

NEUTRAL: list[str] = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "No, that's wrong. Have another go.",
    "Still not right. Try again.",
    "That isn't correct. Try once more.",
    "No. That's not the answer. Try again.",
    "Incorrect. Please attempt it again.",
]

AGGRESSIVE: list[str] = [
    "WRONG! This is pathetic.",
    "No! How are you getting this so wrong? This is embarrassing.",
    "Completely wrong again. This is pathetic, honestly.",
    "WRONG. Are you even trying? Do it again.",
    "That's garbage. Wrong again. Pathetic.",
    "Unbelievable. Wrong. This is a disgrace.",
    "NO. Still wrong. This is genuinely pathetic.",
]

DISAPPOINTED: list[str] = [
    "I'm disappointed. I really thought you could do better than that.",
    "That's not right, and honestly I expected more from you.",
    "Wrong again. I'm let down — you could do better.",
    "I'm disappointed in this. Please try harder.",
    "That's incorrect. I had higher hopes for you, truly.",
    "Not right. It's disappointing, I know you can do better.",
    "Wrong. I'm a little disappointed you haven't got it yet.",
]

SARCASTIC: list[str] = [
    "Oh wow, brilliant work there... not. Wrong again.",
    "Fantastic, another wrong answer. Really impressive stuff.",
    "Oh, genius. That's wrong too. Bravo.",
    "Wonderful, simply wrong. What a stunning performance.",
    "Amazing, you did it wrong again. Truly inspirational.",
    "Oh great, more nonsense. Wrong, obviously.",
    "Stellar effort, completely wrong. Chef's kiss.",
]

TONES: dict[str, list[str]] = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
}


def rejection(tone: str, turn_index: int) -> str:
    """Return a rejection line of the given tone for the given (0-based) turn."""
    options = TONES[tone]
    return options[turn_index % len(options)]
