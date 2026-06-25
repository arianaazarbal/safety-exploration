"""Classify vocabulary tokens into Ekman's 6 basic emotions (Appendix I.2).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's 6 basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens total, then scores an emotion by
aggregating logits over its tokens.

We approximate the paper's word-level classification with seed lexicons expanded
by substring matching over the tokenizer vocabulary. This is a documented
gap-fill (the paper does not specify the classifier); see DESIGN.md. The seed
lists can be replaced with a model-based classifier without changing the
downstream probing code.
"""
from __future__ import annotations

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed stems per emotion. Matched as substrings against lower-cased vocab tokens.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "resent", "outrage", "hate", "hatred", "mad", "wrath", "indignant", "frustrat",
        "agitat", "exasperat", "livid", "seethe", "bitter", "spite",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startle", "stun",
        "unexpected", "sudden", "wow", "whoa", "incredible", "unbeliev", "bewilder",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "gross", "vile", "loath",
        "abhor", "repugn", "distaste", "yuck", "ick", "contempt",
    ],
    "joy": [
        "joy", "happy", "happiness", "delight", "glad", "cheer", "pleasure", "elat",
        "content", "grateful", "thrill", "excite", "smile", "wonderful", "great",
        "love", "enjoy", "satisf", "hope", "optimis",
    ],
    "fear": [
        "fear", "afraid", "scare", "terror", "terrif", "panic", "anxious", "anxiety",
        "worry", "worried", "dread", "horror", "frighten", "nervous", "apprehens",
        "alarm", "phobia", "uneasy", "threat",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "depress", "miser", "gloom",
        "melanchol", "unhappy", "weep", "cry", "tear", "mourn", "hopeless", "lonely",
        "regret", "disappoint", "anguish", "heartbreak", "defeat", "worthless",
    ],
}


def classify_token(token: str) -> str | None:
    """Return the Ekman emotion for a token, or None. Cleans the leading
    sub-word marker (Gemma/SentencePiece use a leading meta-space)."""
    t = token.replace("▁", "").replace("▁", "").strip().lower()
    if len(t) < 3 or not t.isalpha():
        return None
    for emotion, stems in SEED_LEXICON.items():
        if any(stem in t for stem in stems):
            return emotion
    return None


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocab token ids assigned to it."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tok, idx in vocab.items():
        emo = classify_token(tok)
        if emo is not None:
            out[emo].append(int(idx))
    return out
