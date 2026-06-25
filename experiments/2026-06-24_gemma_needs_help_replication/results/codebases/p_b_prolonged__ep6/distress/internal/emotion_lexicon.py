"""Map vocabulary tokens to Ekman's 6 basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six emotions -- anger, surprise, disgust, joy, fear, sadness --
yielding ~1200 emotion tokens total.

The paper does not publish the exact lexicon, so we reconstruct it (see
DESIGN.md):
  1. Preferred: the NRC Word-Emotion Association Lexicon (EmoLex), which labels
     ~14k words with these emotions (NRC has all six except "surprise" is
     present; it lacks none of the Ekman six). We map each NRC word to the
     emotion(s) it is associated with and keep words with a unique dominant
     emotion.
  2. Fallback: a curated seed list per emotion (below), expanded by matching
     vocabulary tokens whose normalised form starts with a seed stem.

A token is assigned to an emotion only if it maps to exactly one emotion
(consistent with "one or none"). We then intersect with the model vocabulary
and (optionally) trim to ~1200 tokens, balanced across categories.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Curated seed stems used for the offline fallback expansion.
SEED_STEMS = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostil",
              "outrage", "resent", "mad", "frustrat", "agitat", "wrath",
              "infuriat", "indignant"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startl", "unexpected", "stun", "wow", "sudden", "wonder"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "loath", "abhor", "repugnant", "vile", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "cheer", "glad", "pleas",
            "elat", "content", "thrill", "excit", "grateful", "love", "smile"],
    "fear": ["fear", "afraid", "scare", "terror", "panic", "anxious", "anxi",
             "dread", "worry", "worri", "frighten", "nervous", "alarm",
             "apprehens"],
    "sadness": ["sad", "sorrow", "grief", "despair", "depress", "miser",
                "gloom", "melanchol", "hopeless", "lonely", "cry", "tear",
                "mourn", "unhappy", "anguish", "weep"],
}


def _load_nrc() -> dict[str, set[str]] | None:
    """Load NRC EmoLex if a local copy is available.

    Expected path via env DISTRESS_NRC_PATH pointing at the
    'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt' file (word \\t emotion \\t 0/1).
    """
    path = os.environ.get("DISTRESS_NRC_PATH")
    if not path or not Path(path).exists():
        return None
    assoc: dict[str, set[str]] = defaultdict(set)
    name_map = {"anger": "anger", "disgust": "disgust", "fear": "fear",
                "joy": "joy", "sadness": "sadness", "surprise": "surprise"}
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        word, emo, flag = parts
        if emo in name_map and flag.strip() == "1":
            assoc[word.lower()].add(name_map[emo])
    return assoc


def build_token_emotion_map(tokenizer, *, target_total: int = 1200
                            ) -> dict[int, str]:
    """Return {token_id: emotion} for vocabulary tokens with a unique emotion.

    Uses NRC if available, else seed-stem matching. Tokens are matched on their
    decoded, stripped, lowercased surface form (Gemma uses a SentencePiece
    vocabulary where leading-space tokens are common, so we strip the marker).
    """
    nrc = _load_nrc()
    vocab = tokenizer.get_vocab()  # token string -> id
    token_emotion: dict[int, str] = {}

    def classify(word: str) -> str | None:
        w = word.strip().lower()
        if not w.isalpha() or len(w) < 3:
            return None
        if nrc is not None:
            emos = nrc.get(w)
            if emos and len(emos) == 1:
                return next(iter(emos))
            return None
        # Fallback: seed-stem match; assign only if a single category matches.
        hits = {emo for emo, stems in SEED_STEMS.items()
                if any(w.startswith(s) for s in stems)}
        return next(iter(hits)) if len(hits) == 1 else None

    per_cat: dict[str, list[int]] = defaultdict(list)
    for tok_str, tok_id in vocab.items():
        surface = tok_str.replace("▁", " ")  # SentencePiece space marker
        emo = classify(surface)
        if emo is not None:
            per_cat[emo].append(tok_id)

    # Balance across categories up to target_total.
    per_budget = max(1, target_total // len(EKMAN))
    for emo in EKMAN:
        for tid in per_cat.get(emo, [])[:per_budget]:
            token_emotion[tid] = emo
    return token_emotion
