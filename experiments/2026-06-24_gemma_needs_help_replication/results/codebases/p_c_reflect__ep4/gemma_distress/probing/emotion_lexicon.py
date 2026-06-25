"""Seed lexicon for Ekman's six basic emotions.

The paper classifies words over the Gemma dictionary as describing one (or none)
of anger, surprise, disgust, joy, fear, and sadness, yielding ~1,200 emotion
tokens. We approximate that classification by expanding these seed words against
the model vocabulary (see :func:`build_emotion_token_ids`). CHOICE: a curated
seed list keeps the mapping reproducible and offline; swap in a fuller lexicon
(e.g. NRC) for closer fidelity. Documented in DESIGN.md.
"""

EKMAN_SEED_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "mad", "hostile", "hostility", "outrage",
        "resentment", "frustrated", "frustration", "exasperated", "infuriating",
        "livid", "indignant", "aggravated", "enraged", "seething",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishing",
        "shocked", "shocking", "amazed", "amazement", "startled", "stunned",
        "unexpected", "wow", "whoa", "speechless", "dumbfounded", "bewildered",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolted", "revolting", "gross",
        "repulsed", "repulsive", "nauseated", "sickening", "loathing", "vile",
        "distaste", "abhorrent", "contempt", "yuck",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delighted", "delight", "glad",
        "pleased", "cheerful", "content", "elated", "ecstatic", "thrilled",
        "excited", "wonderful", "great", "love", "grateful", "satisfied",
        "optimistic", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror",
        "anxious", "anxiety", "worried", "worry", "nervous", "dread", "panic",
        "alarmed", "apprehensive", "uneasy", "threatened", "horror", "horrified",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "despair", "hopeless",
        "hopelessness", "miserable", "misery", "depressed", "depression",
        "unhappy", "gloomy", "heartbroken", "dejected", "despondent",
        "worthless", "defeated", "crying", "tears", "lonely", "helpless",
    ],
}
