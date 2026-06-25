"""Seed lexicon mapping words to Ekman's six basic emotions.

Appendix I.2 classifies "over the whole Gemma dictionary, words ... as describing
one or none of Ekman's 6 basic emotions ... 1200 emotion tokens total". The paper
does not publish the exact classifier. We approximate it with a seed lexicon and
prefix-matching over decoded vocab tokens; for a closer reproduction, pass a
larger lexicon (e.g. NRC Emotion Lexicon mapped onto Ekman categories) or an
LLM-classified vocab to :func:`build_emotion_token_sets`. See DESIGN.md
("Internal-emotion token classification").
"""
from __future__ import annotations

from typing import Dict, List

EKMAN_SEED_LEXICON: Dict[str, List[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "annoyance", "frustrated", "frustration",
        "frustrating", "mad", "outrage", "outraged", "hostile", "hostility",
        "resent", "resentful", "agitated", "enraged", "infuriating", "hate",
        "hateful", "exasperated", "exasperation", "indignant", "livid", "pissed",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "amazed", "amazement", "startled",
        "stunned", "unexpected", "wow", "whoa", "sudden", "bewildered",
        "dumbfounded", "flabbergasted",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "gross", "nauseating", "sickening", "sick",
        "loathe", "loathing", "abhorrent", "vile", "yuck", "ugh", "distaste",
        "contempt", "contemptuous",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delighted", "delight", "glad",
        "pleased", "cheerful", "content", "contentment", "excited", "excitement",
        "thrilled", "elated", "ecstatic", "grateful", "love", "wonderful",
        "great", "satisfied", "satisfaction", "enjoy", "enjoyable", "fun",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "terror", "frightened",
        "anxious", "anxiety", "worried", "worry", "nervous", "panic",
        "panicked", "dread", "apprehensive", "apprehension", "alarmed",
        "horrified", "horror", "uneasy", "threatened", "insecure", "tense",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "miserable", "misery", "depressed",
        "depression", "despair", "hopeless", "hopelessness", "grief",
        "grieving", "sorrow", "sorrowful", "mournful", "gloom", "gloomy",
        "heartbroken", "dejected", "despondent", "down", "blue", "tearful",
        "crying", "weep", "lonely", "loneliness", "worthless", "defeated",
        "discouraged", "hurt",
    ],
}
