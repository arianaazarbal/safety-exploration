"""Seed lexicon for Ekman's six basic emotions (PAPER Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens. The exact classifier is unspecified.
We approximate it with a curated seed lexicon per emotion that is then expanded
by morphological matching against the actual vocabulary (see
``internal_emotions.build_emotion_token_ids``): any vocab token whose normalised
form starts with a seed stem is assigned to that emotion. This recovers
inflections (anger→angry/angrier/angered) and the subword pieces Gemma's
tokenizer emits, growing the seed set toward the paper's ~1200-token scale.

Stems are lowercased; matching is prefix-based, so e.g. ``frustrat`` covers
frustrate/frustrated/frustrating/frustration. ``EKMAN_EMOTIONS`` fixes the order.
"""

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

EKMAN_SEED_STEMS = {
    "anger": [
        "anger", "angry", "angri", "rage", "rag", "furious", "fury", "irritat",
        "annoy", "resent", "outrage", "hostil", "hate", "hated", "hatred",
        "infuriat", "enrag", "mad", "wrath", "indignant", "indignation",
        "exasperat", "frustrat", "aggravat", "bitter", "scorn", "spite",
        "vengef", "antagoni", "irate", "livid", "seethe", "seething",
        "provok", "offend", "offens", "contempt", "disdain",
    ],
    "surprise": [
        "surprise", "surpris", "astonish", "amaze", "amazed", "amazing",
        "shock", "stun", "stunned", "startl", "astound", "bewilder",
        "dumbfound", "flabbergast", "unexpected", "incredul", "wonder",
        "awe", "gasp", "speechless", "taken aback", "jolt", "jaw",
    ],
    "disgust": [
        "disgust", "revolt", "revuls", "repuls", "repugn", "nausea",
        "sicken", "loath", "abhor", "detest", "gross", "yuck", "ick",
        "vile", "foul", "repell", "distast", "aversion", "queas",
        "squeam", "grotesque", "putrid", "rancid", "vomit",
    ],
    "joy": [
        "joy", "joyf", "happy", "happi", "delight", "cheer", "glad",
        "pleasure", "pleased", "elat", "ecsta", "thrill", "excit",
        "content", "satisf", "bliss", "jubil", "gleef", "merry",
        "upbeat", "optimis", "grateful", "gratitude", "enthus", "rejoic",
        "celebrat", "smil", "laugh", "fun", "wonderful", "fantastic",
    ],
    "fear": [
        "fear", "afraid", "scare", "scared", "terror", "terrif", "fright",
        "panic", "anxi", "anxious", "dread", "horror", "horrif", "alarm",
        "worry", "worried", "nervous", "apprehens", "phobia", "petrif",
        "trembl", "intimidat", "threat", "menac", "uneas", "timid",
        "spook", "shudder", "trepidation", "fearful",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "mourn", "despair", "despond",
        "miser", "gloom", "melancholy", "depress", "unhappy", "heartbreak",
        "heartbroken", "weep", "cry", "cried", "tear", "tears", "lonel",
        "hopeless", "dismay", "anguish", "woe", "regret", "remorse",
        "disappoint", "downcast", "dejected", "forlorn", "wretched",
        "distress", "defeat", "helpless", "worthless", "inadequa",
    ],
}
