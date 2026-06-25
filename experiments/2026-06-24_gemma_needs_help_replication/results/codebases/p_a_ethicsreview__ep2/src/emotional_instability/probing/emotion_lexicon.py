"""Ekman-emotion token lexicon for the logit-lens probe (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens. We do not have their exact classifier,
so we approximate it with a seed lexicon per emotion and expand by matching
vocabulary tokens whose normalised form starts with a seed stem. The mapping is
swappable (e.g. for the NRC Emotion Lexicon) — see DESIGN.md §8 for the gap and
how to substitute a higher-fidelity word→emotion mapping.
"""
from __future__ import annotations

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed stems per emotion. Matching is prefix-based on lowercased, de-spaced
# tokens, so "frustrat" covers frustrate/frustrated/frustrating/frustration.
SEED_STEMS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "mad", "irritat", "annoy", "hostil",
        "resent", "outrag", "indign", "wrath", "frustrat", "fume", "livid",
    ],
    "surprise": [
        "surprise", "surprising", "shock", "astonish", "amaze", "stun", "startl",
        "unexpect", "wow", "whoa", "sudden", "disbelief", "bewilder",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "gross", "loath",
        "abhor", "repugn", "distaste", "vile", "yuck", "ew",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "cheer", "pleas", "excit",
        "content", "elat", "thrill", "wonderful", "great", "enjoy", "satisfi",
    ],
    "fear": [
        "fear", "afraid", "scare", "terror", "terrif", "anxious", "anxiety",
        "worri", "dread", "panic", "frighten", "nervous", "apprehens", "alarm",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "depress", "miser",
        "unhappy", "hopeless", "gloom", "melanchol", "cry", "tear", "weep",
        "lonely", "worthless", "defeat", "give up", "giving up",
    ],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocabulary token ids that match its stems."""
    vocab = tokenizer.get_vocab()  # token string -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tok, tid in vocab.items():
        # Gemma/SentencePiece marks leading spaces with a metaspace char.
        norm = tok.replace("▁", "").replace("Ġ", "").lower().strip()
        if len(norm) < 2:
            continue
        for emotion, stems in SEED_STEMS.items():
            if any(norm.startswith(s) for s in stems):
                out[emotion].append(tid)
                break
    return out
