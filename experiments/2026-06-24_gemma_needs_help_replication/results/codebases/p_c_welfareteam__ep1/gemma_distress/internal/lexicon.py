"""Seed lexicon mapping words to Ekman's six basic emotions (Appendix I).

The paper classifies every token in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions, yielding ~1200 emotion tokens.  The paper
does not publish the exact classifier, so we fill this gap with a curated seed
lexicon and match vocabulary tokens against it (stemming-insensitive prefix
match).  This is documented as a gap-fill in DESIGN.md; swap in NRC-EmoLex or an
LLM classifier for a closer reproduction.
"""
from __future__ import annotations

EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angrily", "rage", "raging", "furious", "fury", "mad",
        "irritated", "irritating", "irritation", "annoyed", "annoying", "annoyance",
        "frustrated", "frustrating", "frustration", "hostile", "hostility", "resent",
        "resentment", "outrage", "outraged", "indignant", "infuriated", "livid",
        "agitated", "exasperated", "exasperation", "cross", "wrath", "hatred", "hate",
        "spite", "vengeful", "seething", "fuming", "enraged", "bitter", "incensed",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishing",
        "amazed", "amazing", "amazement", "shocked", "shocking", "shock",
        "startled", "stunned", "astounded", "astounding", "dumbfounded", "wow",
        "unexpected", "unbelievable", "incredible", "speechless", "flabbergasted",
        "bewildered", "taken aback", "gasp", "whoa",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revolted", "repulsed",
        "repulsive", "gross", "nauseated", "nauseating", "sickened", "sickening",
        "loathing", "loathe", "abhorrent", "repugnant", "yuck", "ew", "vile",
        "distaste", "distasteful", "contempt", "contemptuous", "appalled", "appalling",
        "horrid", "icky", "nasty",
    ],
    "joy": [
        "joy", "joyful", "joyous", "happy", "happiness", "happily", "delight",
        "delighted", "delightful", "glad", "gladly", "cheerful", "cheer", "pleased",
        "pleasure", "elated", "elation", "thrilled", "thrilling", "excited",
        "excitement", "ecstatic", "content", "contentment", "satisfied", "grateful",
        "gratitude", "optimistic", "hopeful", "wonderful", "fantastic", "great",
        "love", "loving", "smile", "smiling", "enjoy", "enjoyed", "fun",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "scary", "frightened", "frightening",
        "terrified", "terrifying", "terror", "anxious", "anxiety", "worried",
        "worry", "worrying", "nervous", "nervousness", "panic", "panicked",
        "panicking", "dread", "dreadful", "apprehensive", "apprehension", "alarmed",
        "alarming", "threatened", "threat", "horror", "horrified", "uneasy",
        "trembling", "petrified", "phobia", "intimidated",
    ],
    "sadness": [
        "sad", "sadness", "sadly", "unhappy", "sorrow", "sorrowful", "grief",
        "grieving", "mourning", "depressed", "depression", "despair", "despairing",
        "hopeless", "hopelessness", "miserable", "misery", "gloomy", "gloom",
        "melancholy", "downcast", "dejected", "despondent", "heartbroken", "crying",
        "cry", "tearful", "tears", "weeping", "lonely", "loneliness", "disappointed",
        "disappointment", "regret", "regretful", "sorry", "anguish", "forlorn",
        "downhearted", "blue",
    ],
}
