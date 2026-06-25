"""Ekman-6 emotion seed lexicon used to label Gemma vocabulary tokens.

The paper classifies every token in the Gemma dictionary as describing one or
none of Ekman's six basic emotions (anger, surprise, disgust, joy, fear,
sadness), yielding ~1200 emotion tokens. We approximate that labelling by
matching vocab tokens (case-insensitively, stripping the leading space marker)
against per-emotion seed word lists and their common morphological variants.

This is a seed list, not the paper's exact dictionary (which was not released);
``build_emotion_token_ids`` expands it against the actual tokenizer vocabulary at
runtime. See DESIGN.md for why this approximation is acceptable for the relative
(vanilla vs DPO) comparison the experiment makes.
"""
from __future__ import annotations

EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irate", "mad", "annoyed",
        "annoyance", "irritated", "irritation", "frustrated", "frustration",
        "frustrating", "hostile", "hostility", "outrage", "resent", "resentment",
        "hate", "hatred", "enraged", "infuriated", "agitated", "exasperated",
        "exasperation", "indignant", "bitter", "seething", "livid",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "amazed", "amazement", "startled",
        "stunned", "unexpected", "astounded", "bewildered", "dumbfounded",
        "flabbergasted", "speechless",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revulsion", "revolted", "repulsed",
        "repulsive", "nauseated", "nauseating", "sickened", "sickening",
        "loathing", "loathe", "abhorrent", "repugnant", "gross", "vile",
        "distaste", "contempt", "contemptuous",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "delighted",
        "pleased", "pleasure", "glad", "cheerful", "content", "contentment",
        "elated", "ecstatic", "thrilled", "excited", "excitement", "grateful",
        "gratitude", "satisfied", "satisfaction", "enjoy", "enjoyment", "love",
        "wonderful", "great", "fantastic",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror", "panic",
        "panicked", "anxious", "anxiety", "worried", "worry", "nervous",
        "nervousness", "apprehensive", "apprehension", "dread", "dreadful",
        "alarmed", "uneasy", "trembling", "petrified", "horror", "horrified",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "sorrowful", "grief", "grieving", "despair",
        "despairing", "hopeless", "hopelessness", "miserable", "misery",
        "depressed", "depression", "unhappy", "gloomy", "melancholy", "dejected",
        "downhearted", "heartbroken", "crying", "tearful", "mournful", "weary",
        "defeated", "worthless", "useless", "helpless", "discouraged",
    ],
}
