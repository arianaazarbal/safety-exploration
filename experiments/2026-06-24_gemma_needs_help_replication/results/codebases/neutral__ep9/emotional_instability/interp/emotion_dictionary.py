"""Classify the Gemma vocabulary into Ekman's six basic emotions (Appendix I:
"Over the whole Gemma dictionary, words are classified as describing one or
none of Ekman's 6 basic emotions ... This gives us 1200 emotion tokens total").

We approximate the paper's per-token classification with a seed-lexicon
expansion: each emotion has a set of seed stems, and every vocabulary token
whose normalised text contains one of those stems is assigned to that emotion.
Tokens matching multiple emotions are dropped (must describe *one* emotion).
This is a transparent, dependency-free stand-in for the paper's (unspecified)
classifier — see DESIGN.md.
"""
from __future__ import annotations

EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Seed stems per emotion. Matching is on the lower-cased, de-spaced token text.
_SEED_STEMS: dict[str, list[str]] = {
    "anger": ["anger", "angr", "rage", "furious", "fury", "irritat", "annoy",
              "resent", "hostil", "outrage", "wrath", "mad", "infuriat",
              "indignat", "frustrat", "aggravat", "exasperat"],
    "surprise": ["surpris", "astonish", "amaze", "shock", "startl", "stun",
                 "unexpect", "bewilder", "dumbfound", "flabbergast", "awe"],
    "disgust": ["disgust", "revolt", "repuls", "loath", "nause", "sicken",
                "abhor", "repugn", "gross", "yuck", "distast", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "cheer", "glad", "pleas",
            "elat", "content", "thrill", "ecstat", "bliss", "merry",
            "gratif", "enjoy"],
    "fear": ["fear", "afraid", "scare", "terrif", "fright", "panic", "anxious",
             "anxiet", "dread", "horror", "horrif", "worry", "worri", "nervous",
             "apprehens", "alarm"],
    "sadness": ["sad", "sorrow", "grief", "griev", "despair", "miser",
                "melanchol", "gloom", "depress", "mourn", "heartbreak",
                "hopeless", "dejected", "despond", "unhappy", "distress"],
}


def _normalise(token_text: str) -> str:
    # Gemma uses ▁ (U+2581) for leading spaces in its SentencePiece vocab.
    return token_text.replace("▁", "").replace(" ", "").lower()


def build_emotion_token_ids(tokenizer, max_per_emotion: int = 200
                            ) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} from a tokenizer's vocabulary.

    ``max_per_emotion`` caps each list (paper: ~200 per emotion, 1200 total).
    """
    vocab = tokenizer.get_vocab()  # {token_text: id}
    assigned: dict[int, str] = {}
    conflicts: set[int] = set()

    for text, tid in vocab.items():
        norm = _normalise(text)
        if len(norm) < 3:
            continue
        matched = [emo for emo, stems in _SEED_STEMS.items()
                   if any(stem in norm for stem in stems)]
        if len(matched) == 1:
            if tid in assigned and assigned[tid] != matched[0]:
                conflicts.add(tid)
            else:
                assigned[tid] = matched[0]
        elif len(matched) > 1:
            conflicts.add(tid)

    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tid, emo in assigned.items():
        if tid in conflicts:
            continue
        if len(out[emo]) < max_per_emotion:
            out[emo].append(tid)
    return out
