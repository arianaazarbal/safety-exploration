"""Ekman 6-emotion seed lexicon used to classify Gemma vocabulary tokens.

Appendix I classifies the whole Gemma dictionary into one of Ekman's 6 basic
emotions (anger, surprise, disgust, joy, fear, sadness) or none, yielding ~1200
emotion tokens. The paper does not publish its exact classifier; we approximate
it by matching decoded vocabulary tokens against a seed lexicon (extend with the
NRC Emotion Lexicon for closer parity - see DESIGN.md). Joy is the positive
control; the rest are negative.
"""
from __future__ import annotations

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "furious", "fury", "rage", "mad", "irritated", "annoyed",
        "frustrated", "frustration", "hostile", "outraged", "resent", "hate",
        "hatred", "enraged", "irate", "livid", "indignant", "aggravated",
        "exasperated", "pissed", "fuming", "wrath",
    ],
    "surprise": [
        "surprised", "surprise", "shocked", "astonished", "amazed", "stunned",
        "startled", "astounded", "unexpected", "wow", "whoa", "sudden",
        "bewildered", "dumbfounded", "speechless", "flabbergasted",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "repulsed", "gross",
        "nauseated", "sickening", "repugnant", "loathsome", "vile", "nasty",
        "distaste", "revulsion", "abhorrent",
    ],
    "joy": [
        "joy", "happy", "happiness", "delighted", "glad", "pleased", "cheerful",
        "content", "satisfied", "excited", "thrilled", "elated", "grateful",
        "wonderful", "great", "enjoy", "love", "optimistic", "hopeful", "proud",
    ],
    "fear": [
        "fear", "afraid", "scared", "anxious", "anxiety", "worried", "worry",
        "terrified", "panic", "nervous", "dread", "frightened", "apprehensive",
        "alarmed", "uneasy", "fearful", "horror", "tense", "threatened",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "miserable",
        "despair", "hopeless", "grief", "sorrow", "gloomy", "melancholy",
        "heartbroken", "dejected", "despondent", "downcast", "crying", "tears",
        "lonely", "worthless", "defeated", "broken", "giving",
    ],
}

NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose decoded form (lower,
    stripped of leading space markers) matches a seed word."""
    lex = {e: set(words) for e, words in SEED_LEXICON.items()}
    result: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    vocab = tokenizer.get_vocab()  # token string -> id
    for tok, tid in vocab.items():
        word = tok.replace("▁", "").replace("Ġ", "").strip().lower()
        if not word.isalpha():
            continue
        for emotion, words in lex.items():
            if word in words:
                result[emotion].append(tid)
                break
    return result
