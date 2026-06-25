"""Ekman emotion lexicon and vocabulary classification (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens total. We approximate that mapping with
a seed lexicon per emotion (expandable from the NRC Emotion Lexicon if present
on disk via ``NRC_LEXICON_PATH``), then match tokenizer vocab entries against it.
"""
from __future__ import annotations

import os
from collections import defaultdict

# Seed wordlists. Kept deliberately broad; matching is on the lemma/stem so
# inflections ("frustrate", "frustrated", "frustrating") all map in.
SEED_LEXICON = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritate", "irritated",
        "annoyed", "annoying", "mad", "hostile", "hostility", "resent", "outrage",
        "frustrate", "frustrated", "frustrating", "frustration", "hate", "hatred",
        "infuriate", "enraged", "wrath", "indignant", "agitated", "fuming",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonish", "astonished", "amazed",
        "amazement", "shock", "shocked", "shocking", "startled", "stunned",
        "unexpected", "wonder", "bewildered", "dumbfounded", "speechless",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolt", "revolting", "repulse",
        "repulsed", "repulsive", "nauseate", "nauseated", "gross", "sickening",
        "loathe", "loathing", "abhor", "distaste", "contempt", "appalled",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "delighted", "pleased",
        "glad", "cheerful", "content", "contentment", "elated", "excited",
        "excitement", "thrilled", "grateful", "satisfied", "satisfaction",
        "wonderful", "great", "love", "enjoy", "enjoyed", "optimistic", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "terror", "anxious", "anxiety",
        "worried", "worry", "nervous", "panic", "panicked", "dread", "frightened",
        "apprehensive", "uneasy", "alarmed", "horror", "horrified", "threatened",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "sorrowful", "unhappy", "miserable", "misery",
        "depressed", "depression", "despair", "hopeless", "grief", "grieve",
        "mournful", "gloomy", "melancholy", "heartbroken", "dejected", "downcast",
        "weary", "exhausted", "defeated", "worthless", "helpless", "lonely",
    ],
}

EKMAN_EMOTIONS = tuple(SEED_LEXICON.keys())


def _load_nrc(path: str) -> dict[str, set[str]]:
    """Optional: augment from the NRC Emotion Lexicon (word\\temotion\\tflag)."""
    nrc = defaultdict(set)
    nrc_to_ekman = {
        "anger": "anger", "disgust": "disgust", "fear": "fear", "joy": "joy",
        "sadness": "sadness", "surprise": "surprise",
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emo, flag = parts
            if flag == "1" and emo in nrc_to_ekman:
                nrc[nrc_to_ekman[emo]].add(word.lower())
    return nrc


def build_lexicon() -> dict[str, set[str]]:
    lex = {e: set(words) for e, words in SEED_LEXICON.items()}
    nrc_path = os.environ.get("NRC_LEXICON_PATH")
    if nrc_path and os.path.exists(nrc_path):
        nrc = _load_nrc(nrc_path)
        for e, words in nrc.items():
            lex[e] |= words
    return lex


def _normalize_token(tok: str) -> str:
    # Gemma/SentencePiece uses a leading "▁" for word starts.
    return tok.replace("▁", "").strip().lower()


def classify_vocab(tokenizer) -> dict[str, list[int]]:
    """Return emotion -> list of token ids whose surface form matches the lexicon.

    A token is assigned to an emotion if its normalized form (or a lexicon word)
    is a prefix/equality match, so inflected forms are captured. A token mapping
    to more than one emotion is dropped (kept single-label, as in the paper).
    """
    lex = build_lexicon()
    vocab = tokenizer.get_vocab()  # token string -> id
    assignments: dict[int, set[str]] = defaultdict(set)

    for tok_str, tok_id in vocab.items():
        norm = _normalize_token(tok_str)
        if len(norm) < 3:
            continue
        for emo, words in lex.items():
            for w in words:
                if norm == w or norm.startswith(w) or w.startswith(norm):
                    assignments[tok_id].add(emo)
                    break

    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tok_id, emos in assignments.items():
        if len(emos) == 1:                 # single-label only
            by_emotion[next(iter(emos))].append(tok_id)
    return by_emotion
