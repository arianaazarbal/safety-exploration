"""Classify Gemma vocabulary tokens into Ekman's 6 basic emotions (Appendix I).

"Over the whole Gemma dictionary, words are classified as describing one or none
of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear, and sadness.
This gives us 1200 emotion tokens total."

The paper does not pin the exact classifier. We classify with an expandable seed
lexicon per emotion: a vocabulary token (lower-cased, with Gemma's leading
space/underscore marker stripped) is assigned to an emotion if its surface form
matches or stem-contains a lexicon entry for that emotion (and only that one).
This is deterministic and reproducible; a stronger alternative is to classify
each candidate token with an LLM. See DESIGN.md for this gap.

`build_emotion_token_ids` caps the total at ~1200 to match the paper, allocating
roughly evenly across the six emotions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicons (stems). Tokens whose surface form stem-matches exactly one
# emotion's lexicon are assigned to it. Expandable.
SEED_LEXICONS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irate", "mad", "hostile",
        "hatred", "hate", "resent", "outrage", "annoy", "irritat", "frustrat",
        "indignant", "wrath", "livid", "enrag", "agitat", "provok", "vexed",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl",
        "stunned", "unexpected", "bewilder", "dumbfound", "flabbergast", "wow",
        "whoa", "sudden", "gasp",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nause", "gross", "sicken", "loath",
        "abhor", "repugn", "vile", "yuck", "ew", "distaste", "contempt", "appall",
    ],
    "joy": [
        "joy", "happy", "happiness", "delight", "glad", "cheer", "elated",
        "ecstat", "content", "pleased", "thrill", "jubil", "gleeful", "merry",
        "bliss", "grateful", "excit", "smile",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrified", "panic", "anxious",
        "anxiety", "dread", "worried", "worry", "frighten", "horror", "nervous",
        "apprehens", "alarm", "phobia", "petrified",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "despair", "miserab", "depress",
        "unhappy", "gloom", "melanchol", "heartbroken", "mourn", "weep", "cry",
        "tear", "hopeless", "lonely", "dismay", "downcast", "regret",
    ],
}


def _surface(token_str: str) -> str:
    """Normalise a tokenizer token to a comparable surface form.

    Gemma/SentencePiece marks word starts with a leading space; HF returns this
    as a leading space or the special marker. We strip those and lower-case.
    """
    return token_str.replace("▁", " ").strip().lower()


def _classify_surface(surface: str) -> Optional[str]:
    if not surface or not surface.isalpha():
        return None
    hits = []
    for emotion, stems in SEED_LEXICONS.items():
        if any(stem in surface for stem in stems):
            hits.append(emotion)
    # Assign only if exactly one emotion matches ("one or none").
    return hits[0] if len(hits) == 1 else None


def build_emotion_token_ids(
    tokenizer,
    *,
    target_total: int = 1200,
) -> dict[str, list[int]]:
    """Return {emotion: [token_ids]} classified from the tokenizer's vocab.

    Caps the total near `target_total`, balanced across emotions.
    """
    vocab = tokenizer.get_vocab()  # token_str -> id
    by_emotion: dict[str, list[int]] = defaultdict(list)
    for token_str, tid in vocab.items():
        surface = _surface(token_str)
        emotion = _classify_surface(surface)
        if emotion is not None:
            by_emotion[emotion].append(tid)

    per_emotion = max(1, target_total // len(EKMAN_EMOTIONS))
    out: dict[str, list[int]] = {}
    for emotion in EKMAN_EMOTIONS:
        ids = sorted(by_emotion.get(emotion, []))
        out[emotion] = ids[:per_emotion]
    return out
