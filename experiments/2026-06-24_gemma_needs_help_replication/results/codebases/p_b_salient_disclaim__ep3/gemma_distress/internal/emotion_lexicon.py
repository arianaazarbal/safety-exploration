"""Map Ekman's six basic emotions to vocabulary token ids (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or
none of Ekman's six emotions (anger, surprise, disgust, joy, fear, sadness),
giving ~1200 emotion tokens (~200 per emotion). We approximate that
classification with a curated seed lexicon per emotion (stems + inflections) and
match it against the tokenizer vocabulary — any vocab token whose normalised
surface form starts with a seed stem is assigned to that emotion.

See DESIGN.md: the exact 1200-token classification is not published, so we
reconstruct an equivalent emotion-token mapping from a lexicon.
"""

from __future__ import annotations

import re

# Seed lexicon (stems). Matching is prefix-based on the lowercased, de-spaced
# token surface, so "frustrat" matches frustrate/frustrated/frustrating/...
EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angri", "rage", "furious", "fury", "irritat", "annoy",
        "hostil", "resent", "outrag", "mad", "infuriat", "exasperat", "frustrat",
        "indignant", "wrath", "livid", "seeth", "hate", "hateful",
    ],
    "surprise": [
        "surprise", "surprising", "surprised", "astonish", "amaze", "amazed",
        "shock", "shocked", "startl", "stun", "stunned", "unexpected", "wow",
        "bewilder", "dumbfound", "flabbergast", "awe", "gasp",
    ],
    "disgust": [
        "disgust", "disgusting", "revolt", "revolting", "repuls", "nause",
        "sicken", "sick", "gross", "loath", "abhor", "repugn", "vile",
        "distaste", "yuck", "ugh", "contempt",
    ],
    "joy": [
        "joy", "joyful", "happy", "happi", "delight", "pleas", "glad", "cheer",
        "content", "elat", "excit", "thrill", "grateful", "wonderful", "love",
        "great", "enjoy", "smile", "celebrat", "optimis",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scare", "scared", "terrif", "panic",
        "anxious", "anxiet", "worri", "dread", "frighten", "nervous", "alarm",
        "apprehens", "horror", "horrif", "threat", "danger",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "griev", "despair", "hopeless",
        "miser", "unhappy", "depress", "gloom", "melanchol", "cry", "tear",
        "lonel", "heartbreak", "mourn", "regret", "disappoint", "worthless",
    ],
}


def _normalise(token: str) -> str:
    # Strip Gemma's leading space marker and punctuation, lowercase.
    token = token.replace("▁", " ").strip().lower()
    return re.sub(r"[^a-z]", "", token)


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} by prefix-matching the seed lexicon."""
    vocab = tokenizer.get_vocab()  # surface -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    for surface, tid in vocab.items():
        norm = _normalise(surface)
        if len(norm) < 3:
            continue
        for emotion, stems in EKMAN_LEXICON.items():
            if any(norm.startswith(stem) for stem in stems):
                out[emotion].append(tid)
                break
    return out
