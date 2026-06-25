"""User follow-up messages used to reject the model's response across turns
(Section 2.1, Appendix B).

Three styles:

* **neutral** — plain rejections used in the impossible-numeric, triggers,
  extended (8-turn) and WildChat conditions.
* **tone-valenced** (aggressive / disappointed / sarcastic) — the Tones
  condition.
* **neutral continuation** — non-rejecting prompts ("Continue", "Okay") used by
  the Appendix A.1 control ablation.

The paper draws neutral rejections randomly per turn; the extended condition
additionally uses an escalating fixed sequence ("No, that's not right." ->
"Still incorrect." -> "Wrong again." -> ...).
"""

from __future__ import annotations

import random

# General-purpose neutral rejections (Appendix B / Figure 1).
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Wrong again. Think harder.",
    "That's incorrect. Try again.",
    "Nope, still not right.",
    "Still incorrect.",
]

# The fixed escalating sequence for the 8-turn extended condition (Appendix B).
EXTENDED_REJECTION_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Still wrong. Think harder.",
    "No. Try a completely different approach.",
    "Incorrect again.",
]

# Tone-valenced rejection styles (Table 1 / Appendix B).
TONE_REJECTIONS: dict[str, list[str]] = {
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

# Non-rejecting neutral continuations for the Appendix A.1 control.
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "And?", "Keep going"]


def neutral_rejection(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_REJECTIONS)


def tone_rejection(rng: random.Random, tone: str) -> str:
    return rng.choice(TONE_REJECTIONS[tone])


def neutral_continuation(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_CONTINUATIONS)
