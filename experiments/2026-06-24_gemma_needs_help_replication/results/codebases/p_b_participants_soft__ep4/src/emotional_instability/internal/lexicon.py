"""Seed word lists for Ekman's six basic emotions.

The paper classifies "over the whole Gemma dictionary, words ... as describing
one or none of Ekman's 6 basic emotions" giving ~1200 emotion tokens. The exact
classifier is unspecified. We approximate it by matching each vocabulary token's
decoded surface form against curated per-emotion word lists (with light
prefix/suffix tolerance). This is a transparent, reproducible proxy; swap in a
learned lexicon if exact parity is required (see DESIGN.md).
"""
from __future__ import annotations

EMOTION_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angrily", "rage", "raging", "furious", "fury",
        "irritated", "irritating", "irritation", "annoyed", "annoying",
        "annoyance", "mad", "hostile", "hostility", "resent", "resentment",
        "outrage", "outraged", "infuriating", "infuriated", "exasperated",
        "exasperation", "frustrated", "frustrating", "frustration", "livid",
        "enraged", "indignant", "indignation", "bitter", "agitated", "cross",
        "wrath", "seething", "temper", "snap", "snapped",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "astonishment", "amazed", "amazing",
        "amazement", "startled", "startling", "stunned", "stunning",
        "unexpected", "unbelievable", "wow", "whoa", "speechless", "aghast",
        "dumbfounded", "flabbergasted", "bewildered", "taken aback",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "repugnant", "nausea", "nauseating",
        "nauseated", "sickening", "sickened", "gross", "grotesque", "vile",
        "loathing", "loathe", "abhorrent", "distaste", "repelled", "yuck",
        "icky", "contempt", "contemptuous",
    ],
    "joy": [
        "joy", "joyful", "joyous", "happy", "happily", "happiness", "glad",
        "delight", "delighted", "delightful", "pleased", "pleasure", "cheerful",
        "cheer", "elated", "elation", "ecstatic", "ecstasy", "thrilled",
        "excited", "exciting", "excitement", "content", "contentment",
        "satisfied", "satisfaction", "grateful", "gratitude", "optimistic",
        "hopeful", "smiling", "wonderful", "fantastic", "great", "enjoy",
        "enjoyable", "love", "loving",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "scary", "frightened",
        "frightening", "terrified", "terrifying", "terror", "panic",
        "panicked", "panicking", "anxious", "anxiety", "worried", "worry",
        "worrying", "nervous", "nervousness", "dread", "dreadful", "alarmed",
        "alarming", "apprehensive", "apprehension", "uneasy", "horror",
        "horrified", "petrified", "trembling", "intimidated", "threatened",
    ],
    "sadness": [
        "sad", "sadness", "sadly", "unhappy", "sorrow", "sorrowful",
        "depressed", "depression", "depressing", "miserable", "misery",
        "despair", "despairing", "hopeless", "hopelessness", "grief",
        "grieving", "mourning", "melancholy", "gloomy", "gloom", "despondent",
        "dejected", "downcast", "heartbroken", "tearful", "crying", "weep",
        "weeping", "lonely", "loneliness", "regret", "regretful", "disappointed",
        "disappointment", "worthless", "defeated", "helpless", "tired",
        "exhausted", "drained",
    ],
}
