"""A seed Ekman-6 emotion lexicon used to classify vocabulary tokens.

The paper classifies "over the whole Gemma dictionary, words are classified as
describing one or none of Ekman's 6 basic emotions" (anger, surprise, disgust,
joy, fear, sadness), yielding ~1200 emotion tokens (~200 per category). The
paper does not specify the classifier. We ship a compact seed lexicon (a
stand-in; see DESIGN.md) and classify a vocabulary token if its alphabetic,
lowercased surface form appears in the lexicon. The lexicon is intentionally
swappable for an NRC-style resource or an LLM-labelled vocabulary to better match
the paper's ~200-per-category coverage.
"""

from __future__ import annotations

# Ekman's six basic emotions. "surprise" is included per the paper even though it
# is not strictly valenced; the negative-emotion analysis focuses on anger / fear
# / sadness / disgust, with joy as the positive control.
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

EMOTION_LEXICON: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "rage", "furious", "fury", "irate", "mad", "annoyed",
        "annoyance", "irritated", "irritation", "frustrated", "frustration",
        "outrage", "outraged", "hostile", "hostility", "resentment", "resentful",
        "enraged", "infuriated", "agitated", "indignant", "wrath", "cross",
        "seething", "livid", "bitter", "vengeful", "hatred", "hate", "spiteful",
        "exasperated", "exasperation", "aggravated", "incensed", "fuming",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishment",
        "amazed", "amazement", "shocked", "shock", "startled", "stunned",
        "unexpected", "astounded", "dumbfounded", "bewildered", "flabbergasted",
        "wow", "sudden", "abrupt", "incredible", "unbelievable", "wonder",
        "speechless", "taken aback", "jolt",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "gross", "nauseated", "nausea", "sickened",
        "sickening", "loathing", "loathe", "abhorrent", "repugnant", "vile",
        "distaste", "contempt", "contemptuous", "appalled", "appalling", "yuck",
        "queasy", "offensive", "repelled",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "delighted",
        "pleased", "glad", "cheerful", "elated", "ecstatic", "thrilled",
        "content", "contentment", "satisfied", "satisfaction", "excited",
        "excitement", "grateful", "gratitude", "wonderful", "great", "pleasure",
        "blissful", "jubilant", "merry", "upbeat", "optimistic", "hopeful",
        "love", "loving", "enjoy", "enjoyment", "fun", "smile", "smiling",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror",
        "anxious", "anxiety", "worried", "worry", "nervous", "nervousness",
        "panic", "panicked", "dread", "dreadful", "alarmed", "apprehensive",
        "apprehension", "uneasy", "unease", "phobia", "horror", "horrified",
        "petrified", "trembling", "fearful", "intimidated", "threatened",
        "vulnerable", "helpless",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief", "grieving",
        "depressed", "depression", "despair", "despairing", "miserable",
        "misery", "gloomy", "gloom", "melancholy", "downcast", "dejected",
        "despondent", "hopeless", "hopelessness", "heartbroken", "mournful",
        "weeping", "crying", "tearful", "lonely", "loneliness", "regret",
        "disappointed", "disappointment", "hurt", "anguish", "defeated",
        "worthless", "inadequate", "broken", "tired", "exhausted", "drained",
    ],
}


def lexicon_word_set() -> dict[str, set[str]]:
    return {emo: set(words) for emo, words in EMOTION_LEXICON.items()}
