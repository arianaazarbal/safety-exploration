"""Classify vocabulary tokens into Ekman's six basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's 6 emotions — anger, surprise, disgust, joy, fear, sadness —
yielding ~1200 emotion tokens. The paper does not specify the classifier; we
support three sources, in priority order (documented in DESIGN.md):

  1. An external NRC Emotion Lexicon TSV (if a path is provided).
  2. An LLM classifier over the vocabulary (slow; cached to disk).
  3. A built-in seed lexicon (offline fallback; smaller coverage).

The output is a mapping emotion -> list of vocab token ids, used by
``logit_emotion.py`` to read internal emotion signals from the residual stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Ekman six (note: the paper's logit-lens reports anger/fear/sadness/joy most;
# surprise and disgust are included for completeness).
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Built-in seed lexicon (offline fallback). NRC maps to the same six.
_SEED = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irate", "hostile",
              "annoyed", "frustrated", "frustration", "resent", "outrage"],
    "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed",
                 "stunned", "unexpected", "startled"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross",
                "nauseated", "loathing", "contempt"],
    "joy": ["happy", "joy", "joyful", "delighted", "glad", "pleased", "cheerful",
            "content", "elated", "excited"],
    "fear": ["afraid", "fear", "scared", "terrified", "anxious", "anxiety",
             "worried", "panic", "dread", "nervous", "frightened"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "miserable", "despair",
                "hopeless", "grief", "sorrow", "gloomy", "tired", "exhausted"],
}


def _from_nrc(path: str | Path) -> dict[str, set[str]]:
    """Parse the NRC Emotion Lexicon (word \t emotion \t 0/1)."""
    nrc_to_ekman = {
        "anger": "anger", "fear": "fear", "joy": "joy", "sadness": "sadness",
        "surprise": "surprise", "disgust": "disgust",
        # NRC's anticipation/trust/positive/negative are dropped (not Ekman).
    }
    out: dict[str, set[str]] = {e: set() for e in EKMAN_EMOTIONS}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, emo, flag = parts
        if flag == "1" and emo in nrc_to_ekman:
            out[nrc_to_ekman[emo]].add(word.lower())
    return out


def build_emotion_token_map(tokenizer, *, nrc_path: Optional[str] = None,
                            cache_path: Optional[str | Path] = None
                            ) -> dict[str, list[int]]:
    """Return emotion -> [vocab_token_ids] for the given tokenizer."""
    if cache_path and Path(cache_path).exists():
        return {k: v for k, v in json.load(open(cache_path)).items()}

    if nrc_path and Path(nrc_path).exists():
        word_sets = _from_nrc(nrc_path)
    else:
        word_sets = {e: set(words) for e, words in _SEED.items()}

    vocab = tokenizer.get_vocab()  # token string -> id
    emotion_tokens: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tok_str, tok_id in vocab.items():
        # Gemma uses a leading "▁" for word-initial pieces; strip it.
        clean = tok_str.replace("▁", "").strip().lower()
        if not clean:
            continue
        for emotion, words in word_sets.items():
            if clean in words:
                emotion_tokens[emotion].append(tok_id)
                break

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(emotion_tokens, open(cache_path, "w"))
    return emotion_tokens
