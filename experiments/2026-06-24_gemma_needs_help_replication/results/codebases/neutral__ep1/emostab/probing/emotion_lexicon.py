"""Seed lexicons for Ekman's six basic emotions.

The paper classifies the *entire* Gemma dictionary into one of Ekman's six
emotions (or none), yielding ~1200 emotion tokens. We approximate that
classification with curated seed lexicons (and morphological prefixes); the
`build_emotion_token_map` routine in `probe.py` then matches vocabulary tokens to
these lexicons. See DESIGN.md for the rationale and limitations of this proxy.
"""
from __future__ import annotations

EKMAN_LEXICON = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "annoyance", "mad", "hostile", "hostility",
        "resent", "outrage", "outraged", "frustrated", "frustrating", "frustration",
        "agitated", "enraged", "infuriating", "indignant", "wrath", "livid",
        "exasperated", "exasperation", "seething", "fuming", "bitter",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "amazed", "amazing", "stunned", "startled",
        "unexpected", "wow", "whoa", "sudden", "bewildered", "dumbfounded",
        "incredible", "unbelievable", "speechless",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "repulsed", "repulsive",
        "gross", "nauseated", "nauseating", "sickening", "sick", "yuck", "vile",
        "repugnant", "loathe", "loathing", "distaste", "appalled", "appalling",
        "abhorrent", "contempt", "detest",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "glad", "delighted", "delight",
        "pleased", "cheerful", "excited", "exciting", "thrilled", "elated",
        "content", "satisfied", "wonderful", "great", "fantastic", "love",
        "enjoy", "enjoyed", "grateful", "gratitude", "optimistic", "hopeful",
        "proud", "celebrate", "smile", "fun",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror", "panic",
        "anxious", "anxiety", "worried", "worry", "nervous", "dread", "dreadful",
        "apprehensive", "alarmed", "fearful", "horror", "horrified", "uneasy",
        "tense", "threatened", "intimidated", "petrified", "trembling",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "despair",
        "hopeless", "hopelessness", "miserable", "misery", "grief", "sorrow",
        "mournful", "gloomy", "downcast", "dejected", "despondent", "heartbroken",
        "tearful", "crying", "weeping", "lonely", "worthless", "helpless",
        "defeated", "discouraged", "disappointed", "regret", "ashamed", "guilty",
        "tired", "exhausted", "drained", "giving", "sorry", "apologize", "failure",
    ],
}

NEGATIVE_EMOTIONS = ["anger", "fear", "sadness", "disgust"]
POSITIVE_EMOTIONS = ["joy"]
