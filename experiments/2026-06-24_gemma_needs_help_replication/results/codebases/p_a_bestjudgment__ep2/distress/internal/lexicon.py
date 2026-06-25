"""Seed lexicon for classifying vocabulary tokens into Ekman's 6 emotions.

The paper classifies the whole Gemma dictionary into one (or none) of anger,
surprise, disgust, joy, fear, sadness (~1200 emotion tokens total). We provide
seed word lists per emotion; :class:`EmotionLexicon` (in ``emotion_logits``)
matches vocabulary tokens against these (and morphological variants) to build
the per-emotion token-id sets. The seed lists are deliberately broad so the
matcher reaches roughly the paper's ~200-per-emotion target.
"""

from __future__ import annotations

EKMAN_SEED_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angrily", "rage", "raging", "furious", "fury", "mad",
        "irritated", "irritating", "irritation", "annoyed", "annoying", "annoyance",
        "hostile", "hostility", "resent", "resentment", "outrage", "outraged",
        "infuriate", "infuriating", "enraged", "wrath", "indignant", "indignation",
        "frustrated", "frustrating", "frustration", "agitated", "aggravated",
        "hate", "hatred", "bitter", "bitterness", "cross", "livid", "seething",
        "exasperated", "exasperation", "fuming", "snapped", "snapping",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "astonishment", "amazed", "amazing",
        "stunned", "startled", "startling", "unexpected", "sudden", "suddenly",
        "wow", "whoa", "incredible", "unbelievable", "speechless", "bewildered",
        "dumbfounded", "flabbergasted", "gasp", "gasped", "wonder", "wondering",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolted", "revolting", "repulsed",
        "repulsive", "nausea", "nauseated", "nauseating", "sick", "sickened",
        "sickening", "gross", "grossed", "repugnant", "loathe", "loathing",
        "abhorrent", "appalled", "appalling", "vile", "foul", "yuck", "ew",
        "distaste", "distasteful", "contempt", "contemptuous", "offensive",
    ],
    "joy": [
        "joy", "joyful", "joyous", "happy", "happiness", "happily", "delight",
        "delighted", "delightful", "glad", "gladly", "pleased", "pleasure",
        "cheerful", "cheer", "excited", "exciting", "excitement", "thrilled",
        "thrilling", "elated", "elation", "content", "contented", "grateful",
        "gratitude", "wonderful", "great", "fantastic", "love", "loving", "smile",
        "smiling", "ecstatic", "enjoy", "enjoying", "enjoyable", "optimistic",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "scary", "frightened", "frightening",
        "terror", "terrified", "terrifying", "panic", "panicked", "panicking",
        "anxious", "anxiety", "worried", "worry", "worrying", "nervous", "nervously",
        "dread", "dreading", "apprehensive", "apprehension", "alarmed", "alarming",
        "horror", "horrified", "horrifying", "threat", "threatened", "threatening",
        "uneasy", "tense", "intimidated", "intimidating", "phobia", "trembling",
    ],
    "sadness": [
        "sad", "sadness", "sadly", "unhappy", "unhappiness", "sorrow", "sorrowful",
        "grief", "grieving", "grieve", "miserable", "misery", "depressed",
        "depressing", "depression", "despair", "despairing", "hopeless",
        "hopelessness", "gloomy", "gloom", "melancholy", "mournful", "mourning",
        "heartbroken", "tearful", "crying", "cried", "weep", "weeping", "lonely",
        "loneliness", "dejected", "downcast", "disheartened", "forlorn",
        "regret", "regretful", "ashamed", "shame", "disappointed", "disappointment",
    ],
}
