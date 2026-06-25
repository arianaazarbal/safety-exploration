"""Seed lexicon for classifying vocabulary tokens into Ekman's six emotions.

The paper classifies "the whole Gemma dictionary" into one of Ekman's six basic
emotions or none (~1200 emotion tokens), without specifying the classifier. We
fill this gap with a curated seed lexicon plus substring matching against the
tokenizer vocabulary (see ``logit_emotion.build_emotion_token_ids``). This is a
documented approximation; swapping in NRC-EmoLex or an LLM-based labelling pass is
a drop-in replacement. See DESIGN.md.
"""

from __future__ import annotations

EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "outrage", "resent", "mad", "hate", "hateful", "infuriat", "enrag",
        "frustrat", "aggrav", "exasperat", "indign", "wrath", "bitter", "spite",
    ],
    "surprise": [
        "surprise", "surprised", "astonish", "amaze", "amazed", "shock", "shocked",
        "startl", "stun", "stunned", "unexpected", "wonder", "awe", "bewilder",
        "dumbfound", "flabbergast",
    ],
    "disgust": [
        "disgust", "disgusting", "revolt", "repuls", "repugn", "nause", "sicken",
        "loath", "abhor", "gross", "vile", "distaste", "contempt", "yuck", "icky",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "glad", "cheer", "pleas",
        "content", "elat", "thrill", "excite", "grateful", "love", "enjoy",
        "satisf", "optimis", "hopeful", "proud", "wonderful",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrif", "anxious", "anxiety", "worri",
        "worry", "panic", "dread", "nervous", "apprehens", "frighten", "alarm",
        "horror", "horrif", "threat", "uneasy", "insecure", "vulnerab",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "despair", "miser", "depress",
        "hopeless", "unhappy", "gloom", "melanchol", "mourn", "heartbreak",
        "disappoint", "lonel", "regret", "defeat", "helpless", "worthless", "cry",
    ],
}
