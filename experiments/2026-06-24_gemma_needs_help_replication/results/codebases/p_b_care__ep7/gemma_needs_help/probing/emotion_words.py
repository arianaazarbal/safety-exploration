"""Seed word lists for Ekman's six basic emotions.

Used to classify the Gemma vocabulary into emotion-token sets (Appendix I:
"words are classified as describing one or none of Ekman's 6 basic emotions").
A vocabulary token is assigned to an emotion if its lowercased, stripped surface
form (or its stem) matches that emotion's list. These are seed lemmas; the
matcher in internal_emotions.py also accepts simple morphological variants, so
the realised token set is larger (the paper reports ~1200 tokens total).
"""

from __future__ import annotations

EKMAN_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angic", "rage", "furious", "fury", "irritated",
        "irritation", "annoyed", "annoying", "annoyance", "mad", "outrage",
        "outraged", "hostile", "hostility", "resent", "resentment", "frustrate",
        "frustrated", "frustration", "frustrating", "agitated", "enraged",
        "indignant", "irate", "livid", "seething", "exasperated", "exasperation",
        "hate", "hatred", "bitter", "cross", "snap", "snapped", "wrath",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "astonishment", "amazed", "amazement",
        "startled", "stunned", "unexpected", "wow", "whoa", "speechless",
        "dumbfounded", "flabbergasted", "bewildered", "baffled",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "nauseated", "nauseating", "gross", "sick",
        "sickening", "loathe", "loathing", "abhor", "abhorrent", "distaste",
        "contempt", "contemptuous", "yuck", "ugh", "appalling", "appalled",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "glad", "delighted", "delight",
        "pleased", "pleasure", "cheerful", "content", "contented", "elated",
        "ecstatic", "thrilled", "excited", "excitement", "grateful", "wonderful",
        "great", "fantastic", "love", "lovely", "enjoy", "enjoyable", "smile",
        "celebrate", "satisfied", "satisfaction", "optimistic", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scared", "scary", "terrified", "terror", "terrifying",
        "anxious", "anxiety", "worried", "worry", "worrying", "nervous", "panic",
        "panicked", "dread", "dreadful", "frightened", "frightening", "alarmed",
        "apprehensive", "apprehension", "uneasy", "intimidated", "horrified",
        "horror", "petrified", "threatened", "insecure",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief", "grieving",
        "depressed", "depression", "despair", "despairing", "hopeless",
        "hopelessness", "miserable", "misery", "gloomy", "melancholy", "down",
        "dejected", "despondent", "heartbroken", "crying", "cry", "tears",
        "tearful", "weep", "mournful", "sorry", "regret", "disappointed",
        "disappointment", "lonely", "loneliness", "worthless", "defeated",
        "helpless", "give", "giving", "exhausted", "tired", "drained",
    ],
}

# Negative emotions aggregated in the paper's reporting.
NEGATIVE_EMOTIONS = ("anger", "disgust", "fear", "sadness")
