"""Classify Gemma vocabulary tokens into Ekman's six basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or
none of Ekman's six emotions (anger, surprise, disgust, joy, fear, sadness),
yielding ~1200 emotion tokens, and then aggregates logits over the tokens in
each category. It does not state *how* words are classified.

We use the NRC Word-Emotion Association Lexicon (Mohammad & Turney), filtered to
Ekman's six categories, mapped onto vocabulary tokens by their decoded,
lowercased, whitespace-stripped surface form. This is a standard, citable
emotion lexicon and a defensible operationalisation of "words describing an
emotion"; the exact token set will differ from the paper's. If the lexicon is
unavailable, a small built-in seed lexicon is used and the difference is logged.
See DESIGN.md.
"""

from __future__ import annotations

import os
import warnings

from ..config import EKMAN_EMOTIONS

# Minimal seed lexicon (fallback only). Real runs should provide NRC-EmoLex.
_SEED = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outrage", "resent", "frustrated", "frustration"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "unexpected", "sudden"],
    "disgust": ["disgust", "disgusting", "revolting", "gross", "nauseous",
                "repulsed", "sickening", "loathing"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheerful",
            "pleased", "content", "excited", "elated", "grateful"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "panic", "dread", "nervous", "frightened"],
    "sadness": ["sad", "sadness", "sorrow", "grief", "despair", "miserable",
                "hopeless", "depressed", "unhappy", "gloomy", "tired", "crying"],
}


def _load_nrc_words() -> dict[str, set[str]]:
    """Load NRC-EmoLex words per Ekman emotion, or fall back to the seed set."""
    # 1) `nrclex`-style package or an explicit file path via NRC_EMOLEX_PATH.
    path = os.environ.get("NRC_EMOLEX_PATH")
    words: dict[str, set[str]] = {e: set() for e in EKMAN_EMOTIONS}
    if path and os.path.exists(path):
        # NRC format: "word\temotion\t0|1" per line.
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                word, emotion, flag = parts
                if emotion in words and flag == "1":
                    words[emotion].add(word.lower())
        if any(words.values()):
            return words
    warnings.warn(
        "NRC-EmoLex not found (set NRC_EMOLEX_PATH to the lexicon file); using a "
        "small built-in seed lexicon. Internal-emotion token sets will be much "
        "smaller than the paper's ~1200 tokens."
    )
    return {e: set(v) for e, v in _SEED.items()}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose surface form is an
    emotion word. A token matches if its decoded, stripped, lowercased form is in
    that emotion's word set. Tokens matching multiple emotions are dropped
    (the paper assigns each word to at most one emotion).
    """
    words = _load_nrc_words()
    vocab = tokenizer.get_vocab()  # token_str -> id
    per_emotion: dict[str, set[int]] = {e: set() for e in EKMAN_EMOTIONS}
    assigned_count: dict[int, int] = {}

    for tok_str, tid in vocab.items():
        surface = tokenizer.convert_tokens_to_string([tok_str]).strip().lower()
        if not surface or not surface.isalpha():
            continue
        for emotion, wset in words.items():
            if surface in wset:
                per_emotion[emotion].add(tid)
                assigned_count[tid] = assigned_count.get(tid, 0) + 1

    # Drop tokens assigned to >1 emotion (ambiguous).
    return {
        e: sorted(t for t in ids if assigned_count.get(t, 0) == 1)
        for e, ids in per_emotion.items()
    }
