"""Seed lexicon for classifying vocabulary tokens into Ekman's six basic emotions.

Appendix I classifies the entire Gemma dictionary into one (or none) of Ekman's
six emotions, yielding ~1200 emotion tokens. The paper does not publish the exact
classifier. We approximate it with a curated seed lexicon per emotion; the
classifier in emotion_logits.py matches vocabulary tokens (case-insensitively,
ignoring leading whitespace markers like the SentencePiece underscore) against
these stems. See DESIGN.md for the fidelity caveat and an optional LLM-based
classification hook.
"""

# Stems are matched as substrings against the normalised token, so "anger" also
# captures "angered", "angry" (via the "angr" stem), etc. Kept deliberately
# emotion-specific to limit false positives.
EKMAN_LEXICON = {
    "anger": [
        "anger", "angr", "furious", "fury", "rage", "irritat", "annoy", "resent",
        "outrage", "hostil", "hate", "hatred", "mad", "wrath", "indign", "livid",
        "frustrat", "aggravat", "exasperat", "spite", "vengef", "bitter", "scorn",
        "contempt", "loath", "infuriat", "incens", "cross", "snap", "seething",
    ],
    "surprise": [
        "surpris", "astonish", "amaz", "shock", "startl", "stun", "unexpected",
        "wonder", "awe", "dumbfound", "flabbergast", "bewilder", "astound",
        "speechless", "wow", "whoa", "gasp", "marvel", "incredul", "disbelief",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "repugn", "nause", "sicken", "gross",
        "vile", "loathsome", "yuck", "ew", "abhor", "distaste", "repell",
        "offens", "foul", "putrid", "rancid", "queasy", "squeamish", "icky",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glee", "cheer", "elat", "ecstat",
        "content", "pleas", "thrill", "jubil", "bliss", "gratef",
        "satisf", "merry", "upbeat", "optimist", "excit", "love", "smile", "glad",
        "wonderful", "great", "fantastic", "enjoy", "celebrat", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrif", "frighten", "anxi",
        "worry", "worri", "dread", "panic", "horror", "horrif", "nervous",
        "apprehens", "alarm", "phobia", "intimidat", "uneas", "spook", "timid",
        "trembl", "petrif", "threat", "danger", "peril", "menac",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "mourn", "despair", "miser", "gloom",
        "melanchol", "depress", "unhappy", "heartbreak", "weep", "cry", "tear",
        "lonel", "hopeless", "despond", "forlorn", "dismay", "regret", "remorse",
        "anguish", "woe", "downcast", "dejected", "blue", "disappoint", "lost",
    ],
}
