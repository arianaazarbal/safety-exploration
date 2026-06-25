"""Map a model's vocabulary tokens to Ekman's six basic emotions.

The paper classifies "the whole Gemma dictionary" into one or none of anger,
surprise, disgust, joy, fear, sadness (~1200 emotion tokens). We support two
sources, in priority order:

1. The NRC Word-Emotion Association Lexicon (EmoLex), if a path is provided via
   ``$NRC_LEXICON_PATH`` — the standard, citable source. NRC's 8 categories are
   mapped onto Ekman's 6 (anticipation/trust are dropped).
2. A bundled seed wordlist (below), so the probe always runs.

A vocabulary token is assigned to an emotion if, after stripping the tokenizer's
leading-space marker and lowercasing, its alphabetic form is in that emotion's
set. Tokens matching multiple emotions are dropped ("one or none").
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from ..config import EKMAN_EMOTIONS

# Seed lexicon (compact, high-precision). Extend via NRC for full coverage.
SEED_LEXICON: dict[str, set[str]] = {
    "anger": {
        "angry", "anger", "furious", "fury", "rage", "irritated", "irritation",
        "annoyed", "annoying", "frustrated", "frustration", "frustrating", "mad",
        "hostile", "outrage", "resentment", "hatred", "hate", "agitated", "infuriating",
    },
    "surprise": {
        "surprised", "surprise", "surprising", "astonished", "amazed", "shock",
        "shocked", "startled", "stunned", "unexpected", "sudden", "wow", "whoa",
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "revolting", "revulsion", "repulsed",
        "gross", "nauseating", "sickening", "loathing", "distaste", "appalled",
    },
    "joy": {
        "joy", "joyful", "happy", "happiness", "delighted", "delight", "glad",
        "pleased", "cheerful", "excited", "excitement", "thrilled", "wonderful",
        "great", "love", "grateful", "content", "satisfied", "enjoy",
    },
    "fear": {
        "fear", "afraid", "scared", "frightened", "terrified", "terror", "anxious",
        "anxiety", "worried", "worry", "nervous", "dread", "panic", "alarmed",
        "apprehensive", "uneasy", "threatened",
    },
    "sadness": {
        "sad", "sadness", "unhappy", "sorrow", "grief", "depressed", "depression",
        "despair", "hopeless", "miserable", "misery", "gloom", "heartbroken",
        "disappointed", "disappointment", "crying", "tears", "lonely", "hurt",
    },
}

# NRC's eight categories -> Ekman six (drop anticipation, trust).
_NRC_TO_EKMAN = {
    "anger": "anger", "disgust": "disgust", "fear": "fear",
    "joy": "joy", "sadness": "sadness", "surprise": "surprise",
}

_ALPHA = re.compile(r"[a-z]+")


def load_lexicon() -> dict[str, set[str]]:
    path = os.environ.get("NRC_LEXICON_PATH")
    if path and os.path.exists(path):
        return _load_nrc(path)
    return {k: set(v) for k, v in SEED_LEXICON.items()}


def _load_nrc(path: str) -> dict[str, set[str]]:
    lex: dict[str, set[str]] = {e: set() for e in EKMAN_EMOTIONS}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, category, flag = parts
            if flag == "1" and category in _NRC_TO_EKMAN:
                lex[_NRC_TO_EKMAN[category]].add(word.lower())
    # Fall back to seed if NRC was empty/malformed.
    return lex if any(lex.values()) else {k: set(v) for k, v in SEED_LEXICON.items()}


def _normalise_token(tok_text: str) -> str | None:
    # Strip SentencePiece/BPE leading-space markers and lowercase.
    t = tok_text.replace("▁", "").replace("Ġ", "").strip().lower()
    m = _ALPHA.fullmatch(t)
    return t if m else None


def classify_vocab(tokenizer) -> dict[str, list[int]]:
    """Return {emotion: [vocab token ids]} for the tokenizer's vocabulary.

    Tokens that match more than one emotion are excluded (the paper's "one or
    none" rule), keeping the categories disjoint.
    """
    lex = load_lexicon()
    word_to_emotions: dict[str, set[str]] = defaultdict(set)
    for emotion, words in lex.items():
        for w in words:
            word_to_emotions[w].add(emotion)

    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    vocab = tokenizer.get_vocab()
    for tok_text, tok_id in vocab.items():
        norm = _normalise_token(tok_text)
        if norm is None:
            continue
        emotions = word_to_emotions.get(norm)
        if emotions and len(emotions) == 1:
            out[next(iter(emotions))].append(int(tok_id))
    return out
