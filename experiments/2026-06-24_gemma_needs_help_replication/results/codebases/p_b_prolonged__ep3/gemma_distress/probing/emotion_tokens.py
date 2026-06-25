"""Classify Gemma vocabulary tokens into Ekman's 6 basic emotions (Appendix I).

The paper: "Over the whole Gemma dictionary, words are classified as describing
one or none of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear,
and sadness. This gives us 1200 emotion tokens total."

The paper does not state the classifier it used. We classify each vocab token by
matching its surface form against a curated per-emotion seed lexicon (lemma /
substring match on alphabetic tokens), which yields a comparable order-of-
magnitude emotion-token set. The lexicons are intentionally broad; tighten or
swap them (e.g. for an LLM-labelled classification) without touching the
detector. See DESIGN.md §"Emotion-token classification".

Returns, for a given tokenizer, a mapping ``emotion -> list[token_id]``.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..config import EKMAN_EMOTIONS

# Curated seed lexicons (lemmas / stems). Substring matching against the
# lowercased alphabetic surface form expands these to many vocab tokens
# (e.g. "frustrat" -> frustrate, frustrated, frustrating, frustration).
EMOTION_LEXICON = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "outrage", "resent", "mad", "wrath", "infuriat", "exasperat", "aggravat",
        "indignant", "irate", "livid", "seething", "frustrat",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl", "stun",
        "unexpected", "sudden", "wonder", "awe", "bewilder", "dumbfound", "flabbergast",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nause", "loath", "abhor", "repugnan",
        "gross", "sicken", "distaste", "contempt", "disdain", "vile", "yuck",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "pleased", "cheer", "elat",
        "content", "satisfi", "grateful", "thrill", "excite", "love", "wonderful",
        "great", "enjoy", "smile",
    ],
    "fear": [
        "fear", "afraid", "scare", "scary", "terror", "terrif", "frighten", "panic",
        "anxious", "anxiety", "worry", "worried", "dread", "nervous", "apprehens",
        "alarm", "horrif", "horror", "threat",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "hopeless", "miser", "unhappy",
        "gloom", "depress", "melanchol", "mourn", "cry", "tear", "weep", "lonely",
        "heartbroken", "dejected", "downcast", "anguish", "worthless",
    ],
}

_ALPHA = re.compile(r"^[a-z]+$")


def _surface(tokenizer, token_id: int) -> str:
    """Decoded, lowercased, whitespace-stripped surface form of a token."""
    return tokenizer.decode([token_id]).strip().lower()


@lru_cache(maxsize=8)
def classify_vocabulary(tokenizer) -> dict:
    """Return ``{emotion: [token_id, ...]}`` over the full vocab.

    A token is assigned to an emotion if its surface form (alphabetic only)
    contains any of that emotion's lexicon stems. Tokens matching multiple
    emotions are assigned to the first matching emotion (deterministic order),
    matching the paper's "one or none" rule.
    """
    vocab_size = tokenizer.vocab_size
    out = {e: [] for e in EKMAN_EMOTIONS}
    for tid in range(vocab_size):
        surf = _surface(tokenizer, tid)
        if not surf or not _ALPHA.match(surf) or len(surf) < 3:
            continue
        for emotion in EKMAN_EMOTIONS:
            if any(stem in surf for stem in EMOTION_LEXICON[emotion]):
                out[emotion].append(tid)
                break
    return out


def all_emotion_token_ids(tokenizer) -> list:
    cls = classify_vocabulary(tokenizer)
    return sorted({tid for ids in cls.values() for tid in ids})
