"""Classify the model's vocabulary into Ekman's six basic emotions.

The paper classifies "over the whole Gemma dictionary" into one-or-none of anger,
surprise, disgust, joy, fear, sadness (~1200 emotion tokens total). The exact
classifier is unspecified; we use curated seed lexicons per emotion and match
decoded vocabulary tokens (case/space/punctuation-insensitive, with light
morphological matching) against them, capping per emotion to balance the buckets.
A pool of random non-emotion tokens is also returned for the conversation-level
correlation control. See DESIGN.md for this gap-fill.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# Seed lexicons (stems matched as prefixes so e.g. "frustrat" catches
# frustrated/frustrating/frustration).
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "outrage", "resent", "mad", "wrath", "infuriat", "enrag", "indignan",
        "frustrat", "agitat", "hate", "hatred", "livid", "seething", "temper",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl", "stun",
        "unexpected", "sudden", "wow", "whoa", "incredibl", "unbeliev", "wonder",
        "bewilder", "dumbfound", "flabbergast",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "gross", "vile",
        "repugnan", "loath", "abhor", "distaste", "yuck", "icky", "repellen",
        "contempt", "detest",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "pleased", "cheer", "elat",
        "content", "satisf", "thrill", "excit", "wonderful", "great", "love",
        "enjoy", "grateful", "optimis", "hopeful", "smile", "celebrat",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrifi", "anxi", "worry", "worri",
        "panic", "dread", "nervous", "apprehens", "frighten", "horror", "horrifi",
        "alarm", "uneasy", "phobia", "threat", "danger",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "depress", "miser",
        "melanchol", "gloom", "hopeless", "heartbreak", "mourn", "unhappy",
        "downcast", "dejected", "despond", "forlorn", "weep", "cry", "tear",
        "lonely", "regret", "disappoint",
    ],
}

_CLEAN_RE = re.compile(r"[^a-z]")


@dataclass
class EmotionVocab:
    emotion_token_ids: dict[str, list[int]]   # emotion -> vocab token ids
    random_token_ids: list[int]               # control pool
    n_total: int


def _clean_token(s: str) -> str:
    return _CLEAN_RE.sub("", s.lower())


def _matches(word: str, seeds: list[str]) -> bool:
    return any(word.startswith(seed) and len(word) >= len(seed) for seed in seeds)


def build_emotion_vocab(
    tokenizer,
    *,
    per_emotion_cap: int = 200,
    n_random: int = 500,
    min_len: int = 3,
    seed: int = 0,
) -> EmotionVocab:
    """Bucket the tokenizer vocabulary into Ekman emotions + a random control pool."""
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab()  # token string -> id
    emotion_ids: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    assigned: set[int] = set()

    # Sort for determinism.
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    for tok, tid in items:
        word = _clean_token(tok)
        if len(word) < min_len:
            continue
        for emotion, seeds in EKMAN_SEEDS.items():
            if _matches(word, seeds):
                if len(emotion_ids[emotion]) < per_emotion_cap and tid not in assigned:
                    emotion_ids[emotion].append(tid)
                    assigned.add(tid)
                break

    # Random control tokens: alphabetic, not emotion-assigned.
    candidates = [tid for tok, tid in items
                  if len(_clean_token(tok)) >= min_len and tid not in assigned]
    random_ids = rng.sample(candidates, min(n_random, len(candidates)))

    return EmotionVocab(
        emotion_token_ids=emotion_ids,
        random_token_ids=random_ids,
        n_total=sum(len(v) for v in emotion_ids.values()),
    )
