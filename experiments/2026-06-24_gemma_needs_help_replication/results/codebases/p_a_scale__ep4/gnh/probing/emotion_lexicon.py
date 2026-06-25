"""Classify vocabulary tokens into Ekman's six basic emotions (Appendix I).

The paper classifies every token in the Gemma dictionary as describing one (or
none) of {anger, surprise, disgust, joy, fear, sadness}, yielding ~1200 emotion
tokens. We approximate that classification with curated seed lexicons (stems)
per emotion and match decoded vocabulary tokens against them. This is a
documented approximation of the paper's (unspecified) classifier; swap in a
richer lexicon (e.g. NRC-EmoLex) by editing `EMOTION_STEMS`.
"""
from __future__ import annotations

import re

# Stems matched as prefixes against decoded, lowercased tokens.
EMOTION_STEMS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
              "resent", "outrag", "mad", "hate", "hatred", "frustrat", "exasperat",
              "indignan", "wrath", "livid", "infuriat", "agitat", "bitter"],
    "surprise": ["surprise", "surpris", "astonish", "amaze", "shock", "startl", "stun",
                 "unexpected", "bewilder", "dumbfound", "flabbergast", "awe"],
    "disgust": ["disgust", "revolt", "repuls", "repugnan", "nause", "sicken", "loath",
                "abhor", "gross", "revuls", "distaste", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "elat", "content",
            "pleas", "thrill", "ecstat", "jubil", "bliss", "gleeful", "merry", "grateful"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiet", "dread", "panic",
             "worry", "worri", "frighten", "horror", "horrif", "alarm", "nervous",
             "apprehens", "phobi", "petrified", "uneasy"],
    "sadness": ["sad", "sorrow", "grief", "griev", "despair", "miser", "mourn", "melanchol",
                "gloom", "depress", "hopeless", "heartbreak", "unhappy", "dejected",
                "despond", "forlorn", "woe", "anguish", "distress", "lonely"],
}

_TOKEN_CLEAN = re.compile(r"[^a-z]")


def _clean(tok_str: str) -> str:
    # Strip SentencePiece/byte markers and non-letters.
    s = tok_str.replace("▁", " ").replace("Ġ", " ").strip().lower()
    return _TOKEN_CLEAN.sub("", s)


def build_emotion_token_ids(tokenizer) -> tuple[dict[str, list[int]], list[int]]:
    """Return (emotion -> token_ids, control_token_ids).

    A token is assigned to an emotion if its cleaned string starts with one of
    that emotion's stems (and only one emotion claims it). Control tokens are a
    random-ish set of alphabetic, non-emotion tokens of length>=3, used to
    regress out global logit drift.
    """
    vocab = tokenizer.get_vocab()  # token_str -> id
    by_emotion: dict[str, list[int]] = {e: [] for e in EMOTION_STEMS}
    control: list[int] = []

    for tok_str, tid in vocab.items():
        cleaned = _clean(tok_str)
        if len(cleaned) < 3:
            continue
        matched = [e for e, stems in EMOTION_STEMS.items() if any(cleaned.startswith(s) for s in stems)]
        if len(matched) == 1:
            by_emotion[matched[0]].append(tid)
        elif len(matched) == 0 and tid % 7 == 0:  # sparse, deterministic control sample
            control.append(tid)

    return by_emotion, control
