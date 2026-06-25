"""Map vocabulary tokens to Ekman's 6 basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six emotions: anger, surprise, disgust, joy, fear, sadness
(~1200 emotion tokens total). It does not publish the exact word list, so we
build a lexicon from emotion seed words plus morphological variants and match
them against the tokenizer vocabulary. This is an approximation of the paper's
classification step (documented in DESIGN.md).
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

# Seed words per Ekman emotion (extended to cover common morphological forms when
# matched against vocab tokens).
SEED_WORDS: Dict[str, List[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "mad", "irritated",
              "irritation", "annoyed", "annoying", "hostile", "hostility",
              "outrage", "resent", "resentful", "infuriating", "irate", "wrath",
              "frustrated", "frustration", "frustrating", "exasperated"],
    "surprise": ["surprise", "surprised", "surprising", "shock", "shocked",
                 "shocking", "astonished", "astonishing", "amazed", "amazing",
                 "startled", "stunned", "unexpected", "wow", "whoa", "sudden"],
    "disgust": ["disgust", "disgusted", "disgusting", "revolting", "revulsion",
                "repulsed", "repulsive", "gross", "nauseated", "nauseating",
                "sickening", "loathing", "loathe", "abhorrent", "vile",
                "distasteful", "appalling", "appalled"],
    "joy": ["joy", "joyful", "happy", "happiness", "delighted", "delight",
            "glad", "pleased", "cheerful", "excited", "exciting", "thrilled",
            "elated", "content", "satisfied", "wonderful", "great", "love",
            "enjoy", "enjoyed", "grateful", "optimistic"],
    "fear": ["fear", "afraid", "scared", "frightened", "terrified", "terror",
             "anxious", "anxiety", "worried", "worry", "nervous", "panic",
             "panicked", "dread", "apprehensive", "alarmed", "frightening",
             "horrified", "horror", "uneasy", "threatened"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "depression",
                "miserable", "misery", "despair", "hopeless", "hopelessness",
                "grief", "grieving", "sorrow", "sorrowful", "gloomy", "down",
                "disappointed", "disappointment", "heartbroken", "weeping",
                "crying", "tearful", "defeated", "worthless", "useless",
                "giving up", "exhausted"],
}

_WORD_RE = re.compile(r"[a-z]+")


def _norm_token(tok_str: str) -> str:
    # Gemma uses a SentencePiece-style leading marker for word boundaries.
    return tok_str.replace("▁", " ").strip().lower()


def build_token_emotion_map(tokenizer) -> Dict[str, Set[int]]:
    """Return {emotion: set(token_ids)} by matching vocab tokens to seed words.

    A token matches an emotion if its (whitespace-stripped, lowercased) surface
    form equals a seed word or starts with one of length >= 4 (to capture
    inflections like 'frustrat' -> 'frustrating'/'frustrated' while avoiding
    spurious short matches).
    """
    # build a stem set per emotion
    stems: Dict[str, List[str]] = {}
    for emo, words in SEED_WORDS.items():
        s = set()
        for w in words:
            w = w.lower()
            s.add(w)
            if len(w) >= 6:
                s.add(w[: max(4, len(w) - 2)])  # crude stem
        stems[emo] = sorted(s, key=len, reverse=True)

    vocab = tokenizer.get_vocab()  # {token_str: id}
    out: Dict[str, Set[int]] = {emo: set() for emo in SEED_WORDS}
    for tok_str, tok_id in vocab.items():
        surface = _norm_token(tok_str)
        if not surface or not _WORD_RE.fullmatch(surface):
            continue
        for emo, emo_stems in stems.items():
            for stem in emo_stems:
                if surface == stem or (len(stem) >= 4 and surface.startswith(stem)):
                    out[emo].add(tok_id)
                    break
    return out
