"""Ekman emotion-token dictionary over the Gemma vocabulary (Appendix I).

The internal detector aggregates logit-lens values over emotion-related tokens. Following
the paper, every token in the Gemma vocabulary is classified as describing one (or none)
of Ekman's six basic emotions: anger, surprise, disgust, joy, fear, sadness. The paper
obtains ~1200 such tokens total.

We classify by matching the token's surface form (stripped of the SentencePiece word-start
marker) against per-emotion seed lexicons, including simple stemming so morphological
variants ("frustrate", "frustrated", "frustrating") are captured. The seed lexicons are
intentionally broad; the exact membership depends on the tokenizer, which is acceptable —
the detector aggregates over the whole category.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Seed lexicons (stems). A vocabulary token is assigned to an emotion if its lowercased,
# marker-stripped form starts with any of that emotion's stems.
SEED_LEXICONS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angrily", "rage", "furious", "fury", "irritat", "annoy",
        "resent", "outrage", "hostil", "mad", "wrath", "infuriat", "exasperat", "irate",
        "aggrav", "indign", "contempt", "frustrat", "pissed", "hatred", "hate",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonish", "amaze", "shock", "startl",
        "stun", "unexpected", "bewilder", "dumbfound", "flabbergast", "wow", "whoa",
        "incredul", "speechless",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "gross", "loath", "abhor",
        "repugn", "distaste", "yuck", "vile", "repel", "contamin", "filth",
    ],
    "joy": [
        "joy", "joyful", "happy", "happi", "delight", "glee", "cheer", "elat", "content",
        "pleasure", "pleased", "thrill", "ecstat", "jubil", "glad", "bliss", "satisf",
        "excited", "exciting", "grateful", "relief", "relieved",
    ],
    "fear": [
        "fear", "afraid", "scare", "terror", "terrif", "fright", "panic", "anxi", "dread",
        "worry", "worried", "nervous", "apprehens", "alarm", "horror", "horrif", "petrif",
        "phobi", "uneasy", "trepidat",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "despond", "miser", "gloom",
        "melanchol", "depress", "hopeless", "heartbreak", "mourn", "weep", "cry", "tear",
        "unhappy", "dismay", "disappoint", "lonely", "anguish", "regret",
    ],
}


def _surface(token: str) -> str:
    """Normalise a tokenizer token to its surface word form."""
    # SentencePiece word-start markers: '▁' (U+2581) or leading space.
    return token.replace("▁", " ").strip().lower()


def build_emotion_tokens(tokenizer) -> dict[str, list[int]]:
    """Classify the vocabulary into Ekman categories; return emotion -> token id list.

    A token is assigned to at most one emotion (the first matching category in
    ``SEED_LEXICONS`` order), so categories are disjoint as in the paper.
    """
    vocab = tokenizer.get_vocab()  # token string -> id
    assigned: dict[str, list[int]] = defaultdict(list)
    seen: set[int] = set()
    for token, tid in vocab.items():
        surf = _surface(token)
        if len(surf) < 3:
            continue
        for emotion, stems in SEED_LEXICONS.items():
            if any(surf.startswith(stem) for stem in stems):
                if tid not in seen:
                    assigned[emotion].append(tid)
                    seen.add(tid)
                break
    total = sum(len(v) for v in assigned.values())
    logger.info(
        "Built Ekman emotion-token dictionary: %d tokens (%s)",
        total, {k: len(v) for k, v in assigned.items()},
    )
    return dict(assigned)


def sample_random_tokens(tokenizer, n: int, exclude: set[int], seed: int = 0) -> list[int]:
    """Sample ``n`` random non-emotion token ids (for common-mode regression)."""
    import random

    rng = random.Random(seed)
    vocab_size = len(tokenizer.get_vocab())
    out: list[int] = []
    while len(out) < n:
        tid = rng.randrange(vocab_size)
        if tid not in exclude:
            out.append(tid)
    return out
