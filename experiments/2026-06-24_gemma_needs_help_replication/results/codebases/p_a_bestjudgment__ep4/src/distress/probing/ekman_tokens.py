"""Classify vocabulary tokens by Ekman basic emotion (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — giving ~1200 emotion tokens total. We reproduce this by intersecting the
tokenizer vocabulary with an emotion lexicon.

Lexicon source: the NRC Word-Emotion Association Lexicon (EmoLex) if available
locally (set ``NRC_LEXICON`` to its path), mapped onto the six Ekman categories.
NRC has eight categories; we map anger/disgust/fear/joy/sadness/surprise directly
and drop trust/anticipation (not Ekman basic emotions). If NRC is unavailable we
fall back to a curated seed lexicon so the pipeline is runnable offline (smaller
than 1200 tokens; documented in DESIGN.md).
"""

from __future__ import annotations

import os
from collections import defaultdict

EKMAN = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Curated offline fallback seed words per Ekman emotion.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outrage", "resent", "frustrated", "frustration", "hate",
              "fury", "enraged", "infuriating", "livid", "aggravated"],
    "surprise": ["surprised", "surprise", "shock", "shocked", "astonished", "amazed",
                 "stunned", "startled", "unexpected", "wow", "sudden", "astounding"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross", "nauseated",
                "sickened", "loathing", "repugnant", "distaste", "abhorrent"],
    "joy": ["happy", "joy", "joyful", "delighted", "glad", "pleased", "cheerful",
            "excited", "content", "elated", "thrilled", "wonderful", "great", "love"],
    "fear": ["afraid", "fear", "fearful", "scared", "terrified", "anxious", "anxiety",
             "worried", "worry", "nervous", "panic", "dread", "frightened", "apprehensive"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "miserable",
                "hopeless", "grief", "sorrow", "gloomy", "down", "crying", "cry",
                "tearful", "heartbroken", "disappointed", "worthless", "failure"],
}


def _load_nrc() -> dict[str, set[str]] | None:
    path = os.environ.get("NRC_LEXICON")
    if not path or not os.path.exists(path):
        return None
    cats: dict[str, set[str]] = defaultdict(set)
    keep = set(EKMAN)
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion in keep and flag == "1":
                cats[emotion].add(word.lower())
    return cats


def build_lexicon() -> dict[str, set[str]]:
    nrc = _load_nrc()
    if nrc:
        return {e: set(nrc.get(e, set())) for e in EKMAN}
    return {e: set(words) for e, words in SEED_LEXICON.items()}


def classify_vocab_tokens(tokenizer) -> dict[str, list[int]]:
    """Return ``{emotion: [token_id, ...]}`` for tokens whose surface word is in
    the lexicon for exactly one emotion (tokens claimed by multiple emotions are
    dropped to keep categories disjoint)."""
    lexicon = build_lexicon()
    word_to_emotion: dict[str, str] = {}
    seen_multi: set[str] = set()
    for emotion, words in lexicon.items():
        for w in words:
            if w in word_to_emotion and word_to_emotion[w] != emotion:
                seen_multi.add(w)
            else:
                word_to_emotion[w] = emotion
    for w in seen_multi:
        word_to_emotion.pop(w, None)

    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        # Gemma uses a leading "▁" (U+2581) to mark word starts; strip it and punctuation.
        surface = tok.replace("▁", " ").strip().lower()
        if not surface.isalpha():
            continue
        emotion = word_to_emotion.get(surface)
        if emotion:
            out[emotion].append(tid)
    return out
