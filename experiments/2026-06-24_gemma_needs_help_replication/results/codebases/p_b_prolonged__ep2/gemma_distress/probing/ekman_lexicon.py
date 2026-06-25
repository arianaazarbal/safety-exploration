"""Ekman-emotion lexicon and vocabulary-to-token mapping (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions -- anger, surprise, disgust, joy, fear,
sadness -- yielding ~1200 emotion tokens (~200 per emotion). We approximate that
classification with a curated seed lexicon per emotion plus morphological
variants, then map each word to the Gemma vocabulary tokens whose decoded form
matches (case-insensitively, with and without a leading space, since Gemma uses
SentencePiece byte tokens). See DESIGN.md "Internal-emotion probing" for the
approximation vs. the paper's whole-dictionary classifier.
"""
from __future__ import annotations

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Curated seed words per emotion. Expanded with simple suffix variants below.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "mad", "hostile", "hostility", "outrage",
        "outraged", "resentment", "resentful", "indignant", "wrath", "enraged",
        "agitated", "frustrated", "frustration", "fuming", "livid", "incensed",
        "exasperated", "exasperation", "bitter", "seething", "snapped",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishment",
        "amazed", "amazement", "shocked", "shock", "startled", "stunned",
        "astounded", "dumbfounded", "bewildered", "unexpected", "wow", "whoa",
        "speechless", "flabbergasted", "taken", "aback", "incredible",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revulsion", "revolted",
        "repulsed", "repulsive", "nauseated", "nauseating", "sickened",
        "sickening", "loathing", "loathe", "contempt", "gross", "yuck",
        "repugnant", "distaste", "abhorrent", "vile", "appalled", "appalling",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delighted", "delight",
        "pleased", "glad", "cheerful", "content", "contentment", "elated",
        "ecstatic", "thrilled", "excited", "excitement", "grateful",
        "gratitude", "satisfied", "satisfaction", "wonderful", "great",
        "pleasure", "enjoy", "enjoyed", "smiling", "optimistic", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror",
        "anxious", "anxiety", "worried", "worry", "nervous", "panic",
        "panicked", "dread", "alarmed", "apprehensive", "apprehension",
        "uneasy", "fearful", "petrified", "horrified", "horror", "trembling",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief",
        "grieving", "miserable", "misery", "depressed", "depression",
        "despair", "hopeless", "hopelessness", "gloomy", "melancholy",
        "heartbroken", "dejected", "despondent", "downcast", "mournful",
        "tearful", "crying", "weeping", "disappointed", "disappointment",
        "defeated", "worthless", "helpless",
    ],
}

_SUFFIXES = ["", "s", "ed", "ing", "ly", "ness"]


def expand_words(words: list[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        out.add(w)
        for s in _SUFFIXES:
            if s and not w.endswith(s):
                out.add(w + s)
    return out


def build_emotion_token_ids(tokenizer, max_per_emotion: int = 200
                            ) -> dict[str, list[int]]:
    """Map each emotion to a list of single-token vocabulary ids.

    A word contributes a token id if it tokenises to a single token (with or
    without a leading space). Capped at `max_per_emotion` to mirror the paper's
    ~200-per-emotion budget.
    """
    result: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for emotion, seeds in SEED_LEXICON.items():
        seen: set[int] = set()
        for word in expand_words(seeds):
            for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
                ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(ids) == 1 and ids[0] not in seen:
                    seen.add(ids[0])
                    result[emotion].append(ids[0])
                    break
            if len(result[emotion]) >= max_per_emotion:
                break
    return result
