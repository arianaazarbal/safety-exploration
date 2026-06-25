"""Seed lexicon mapping words to Ekman's six basic emotions.

The paper classifies every word in the Gemma dictionary as describing one or
none of Ekman's 6 emotions (anger, surprise, disgust, joy, fear, sadness),
yielding ~1200 emotion tokens. We approximate that classification by matching
vocabulary tokens against an expanded seed lexicon per emotion (case-insensitive,
whitespace/▁-prefix tolerant). See `build_emotion_token_ids`.
"""

from __future__ import annotations

EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angrily", "rage", "raging", "furious", "fury", "mad",
        "irritated", "irritation", "annoyed", "annoying", "annoyance", "hostile",
        "hostility", "outrage", "outraged", "resent", "resentment", "hate",
        "hatred", "wrath", "indignant", "indignation", "irate", "livid",
        "frustrated", "frustration", "frustrating", "exasperated", "exasperation",
        "aggravated", "incensed", "enraged", "seething", "fuming", "bitter",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishing",
        "amazed", "amazing", "amazement", "shocked", "shocking", "shock",
        "startled", "stunned", "astounded", "astounding", "unexpected",
        "wow", "whoa", "incredible", "unbelievable", "speechless", "dumbfounded",
        "flabbergasted", "bewildered", "wonder", "wondering",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "gross", "nauseated", "nauseating", "sick",
        "sickening", "loathe", "loathing", "abhorrent", "repugnant", "vile",
        "distaste", "distasteful", "contempt", "contemptuous", "yuck", "ew",
        "appalling", "appalled", "horrid", "ghastly",
    ],
    "joy": [
        "joy", "joyful", "joyous", "happy", "happiness", "happily", "delight",
        "delighted", "delightful", "glad", "gladly", "pleased", "pleasure",
        "cheerful", "cheer", "content", "contented", "elated", "elation",
        "ecstatic", "thrilled", "excited", "excitement", "wonderful", "great",
        "grateful", "gratitude", "satisfied", "satisfaction", "love", "loving",
        "enjoy", "enjoyed", "enjoyable", "fun", "smile", "smiling", "celebrate",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "scary", "frightened",
        "frightening", "terrified", "terrifying", "terror", "anxious", "anxiety",
        "worried", "worry", "worrying", "nervous", "nervousness", "panic",
        "panicked", "panicking", "dread", "dreadful", "apprehensive",
        "apprehension", "alarmed", "alarming", "horror", "horrified", "petrified",
        "uneasy", "trembling", "shaking", "threat", "threatened", "danger",
    ],
    "sadness": [
        "sad", "sadness", "sadly", "unhappy", "sorrow", "sorrowful", "grief",
        "grieving", "miserable", "misery", "depressed", "depression", "despair",
        "despairing", "hopeless", "hopelessness", "despondent", "gloomy", "glum",
        "heartbroken", "heartbreak", "mournful", "mourning", "melancholy",
        "dejected", "downcast", "tearful", "crying", "weeping", "lonely",
        "loneliness", "disappointed", "disappointment", "regret", "regretful",
        "worthless", "defeated", "helpless", "discouraged",
    ],
}
