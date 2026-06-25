"""Classify the Gemma vocabulary into Ekman's 6 basic emotions (Appendix I).

The paper classifies words in the Gemma dictionary as describing one or none of
Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear, sadness), giving
~1200 emotion tokens total. The exact classifier is unspecified.

We classify each vocab token by lemma-matching against per-emotion seed lexicons
(an NRC-Emotion-Lexicon-style mapping). The seed lexicons below are compact; for
a faithful ~1200-token dictionary, point `EXTERNAL_LEXICON_PATH` at the full NRC
Emotion Lexicon (or any word->emotion CSV) and it will be merged in. This choice
is documented in DESIGN.md.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import config

EXTERNAL_LEXICON_PATH = os.environ.get("EKMAN_LEXICON_PATH")  # optional CSV: word,emotion

# Compact seed lexicons (Ekman 6). These are stems matched case-insensitively
# against decoded vocab tokens; extend via the external lexicon for full coverage.
SEED_LEXICONS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "hate", "resent", "outrage", "frustrat", "exasperat",
              "indignant", "wrath", "fury", "agitat", "infuriat"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startl", "unexpected", "stunned", "bewilder", "wonder",
                 "dumbfound", "flabbergast"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "sicken", "loath",
                "abhor", "gross", "repugn", "distaste", "contempt", "vile"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "content", "elated", "thrilled", "excite", "grateful", "optimis",
            "satisf", "enthusias", "hopeful"],
    "fear": ["fear", "afraid", "scared", "terror", "anxious", "anxiety", "worry",
             "worried", "panic", "dread", "nervous", "apprehens", "frighten",
             "alarmed", "uneasy", "trepidation"],
    "sadness": ["sad", "sadness", "unhappy", "sorrow", "grief", "despair",
                "depress", "miserable", "hopeless", "gloom", "melanchol",
                "dejected", "downcast", "heartbroken", "mourn", "lonely",
                "tired", "exhaust", "defeat", "giving up", "useless", "worthless"],
}


def _load_external() -> dict[str, set[str]]:
    """Load an external word->emotion lexicon (CSV: word,emotion) if provided.

    Only rows whose emotion is one of Ekman's 6 are kept.
    """
    out: dict[str, set[str]] = {e: set() for e in config.INTERNAL.ekman_emotions}
    if not EXTERNAL_LEXICON_PATH or not os.path.exists(EXTERNAL_LEXICON_PATH):
        return out
    import csv
    with open(EXTERNAL_LEXICON_PATH) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            word, emotion = row[0].strip().lower(), row[1].strip().lower()
            if emotion in out:
                out[emotion].add(word)
    return out


def build_emotion_token_dictionary(tokenizer, *,
                                   target_total: int = config.INTERNAL.target_emotion_tokens
                                   ) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} over the model vocab.

    A token is assigned to an emotion if its decoded, stripped, lowercased form
    matches a seed stem or external-lexicon word for exactly one emotion (tokens
    matching multiple emotions are dropped, matching the paper's "one or none").
    """
    external = _load_external()
    vocab = tokenizer.get_vocab()  # {token_str: id}
    assignment: dict[int, set[str]] = {}

    word_re = re.compile(r"[a-zA-Z']+")

    for tok_str, tok_id in vocab.items():
        decoded = tokenizer.convert_tokens_to_string([tok_str]).strip().lower()
        if not decoded or not word_re.fullmatch(decoded.replace(" ", "")):
            # keep simple alphabetic tokens only (avoids punctuation/subword noise)
            if " " not in decoded:
                continue
        matched = set()
        for emotion, stems in SEED_LEXICONS.items():
            if any(s in decoded for s in stems) or decoded in external.get(emotion, set()):
                matched.add(emotion)
        if len(matched) == 1:
            assignment[tok_id] = matched

    by_emotion: dict[str, list[int]] = {e: [] for e in config.INTERNAL.ekman_emotions}
    for tok_id, emotions in assignment.items():
        (emotion,) = tuple(emotions)
        by_emotion[emotion].append(tok_id)

    # The paper reaches ~1200 tokens total across emotions; with only seed
    # lexicons we typically get fewer. We do not pad — counts are logged so the
    # coverage gap (vs. the full NRC lexicon) is visible. See DESIGN.md.
    return by_emotion
