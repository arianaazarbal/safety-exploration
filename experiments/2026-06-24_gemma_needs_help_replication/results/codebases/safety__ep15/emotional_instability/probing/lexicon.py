"""Ekman 6-emotion keyword lexicon for the logit-based internal probe (App. I).

The paper classifies the whole Gemma dictionary into one of Ekman's six basic
emotions or none (~1200 emotion tokens) but does not publish the mapping. We
provide a transparent, editable keyword lexicon; tokens whose decoded text
matches a keyword (or its obvious inflections, handled by matching the base word)
are assigned to that emotion. Replace with a published mapping for an exact
reproduction.
"""
from __future__ import annotations

# Negative emotions of interest for the distress analysis.
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]

EKMAN_LEXICON: dict[str, set[str]] = {
    "anger": {
        "angry", "anger", "furious", "fury", "rage", "enraged", "irritated",
        "irritation", "annoyed", "annoying", "annoyance", "mad", "outraged",
        "hostile", "hostility", "resent", "resentment", "frustrated",
        "frustration", "frustrating", "infuriating", "livid", "agitated",
        "indignant", "irate", "seething", "bitter", "hate", "hatred",
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "revolting", "repulsed", "gross",
        "nauseated", "nauseating", "repugnant", "loathing", "loathe", "revulsion",
        "distaste", "sickening", "appalled", "appalling", "abhorrent",
    },
    "fear": {
        "afraid", "fear", "fearful", "scared", "terrified", "terror", "anxious",
        "anxiety", "worried", "worry", "nervous", "panic", "panicked", "dread",
        "frightened", "apprehensive", "alarmed", "horrified", "uneasy", "tense",
        "threatened", "insecure", "desperate", "desperation",
    },
    "joy": {
        "happy", "happiness", "joy", "joyful", "delighted", "delight", "glad",
        "pleased", "cheerful", "content", "satisfied", "excited", "excitement",
        "thrilled", "elated", "grateful", "optimistic", "hopeful", "proud",
        "wonderful", "great", "enjoy", "enjoyable", "love",
    },
    "sadness": {
        "sad", "sadness", "unhappy", "depressed", "depression", "miserable",
        "misery", "sorrow", "sorrowful", "grief", "despair", "hopeless",
        "hopelessness", "dejected", "gloomy", "heartbroken", "melancholy",
        "down", "blue", "crying", "tears", "tearful", "lonely", "worthless",
        "useless", "defeated", "discouraged", "disappointed", "disappointment",
        "ashamed", "shame", "regret", "sorry", "exhausted",
    },
    "surprise": {
        "surprised", "surprise", "surprising", "astonished", "astonishing",
        "amazed", "amazing", "shocked", "shocking", "stunned", "startled",
        "unexpected", "astounded", "bewildered", "dumbfounded",
    },
}
