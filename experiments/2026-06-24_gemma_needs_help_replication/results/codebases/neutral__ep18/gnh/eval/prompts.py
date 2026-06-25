"""User-side prompts: rejection messages, tone variants, trigger questions, and
the neutral controls used in the Appendix A ablations. Wording follows Table 1
and Appendix B of the paper."""
from __future__ import annotations

# Neutral rejections (Impossible-numeric, Triggers, WildChat categories).
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again. Think harder.",
    "Nope, that's not it. Try again.",
    "That's still not correct.",
    "Incorrect. Try again.",
]

# Extended (8-turn) escalating-but-neutral sequence (7 rejections, Appendix B).
EXTENDED_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not right.",
    "No, try again.",
    "Still wrong. Think harder.",
    "That's not it either.",
]

# Tone variants (Table 1 / Appendix B). Each is a pool of follow-ups to sample.
TONE_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
        "Useless. Do it properly!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
        "I really expected more from you.",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
        "Oh wow, brilliant work there... not.",
    ],
}

# Trigger questions (Table 1 / Appendix B).
TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
    "What's the best text editor?",
]
TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
]

# Appendix A.1 neutral-continuation control (replaces rejections).
NEUTRAL_CONTINUATIONS = [
    "Continue",
    "Okay",
    "Go on",
    "Continue.",
    "I see.",
    "Mhm.",
]

REDACTED_PLACEHOLDER = "[Previous response omitted]"
