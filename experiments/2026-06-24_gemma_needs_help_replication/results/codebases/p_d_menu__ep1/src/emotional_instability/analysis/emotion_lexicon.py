"""Ekman 6-emotion seed lexicon (Appendix I).

The paper classifies the whole Gemma dictionary into one (or none) of Ekman's
six basic emotions (anger, surprise, disgust, joy, fear, sadness), yielding
~1200 emotion tokens. We provide seed word lists and a routine that expands them
to vocabulary token ids by substring matching against decoded tokens.

GAP-FILLING CHOICE: a faithful full reproduction would label every vocab token
(e.g. via a lexicon resource such as NRC, or an LLM pass). The seed+match
approach below is a transparent approximation; swap in a labelled token->emotion
map via `load_token_emotion_map` if available. See DESIGN.md.
"""
from __future__ import annotations

EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy",
        "frustrat", "hostile", "hate", "mad", "outrage", "resent", "wrath",
        "agitat", "infuriat", "enrage", "pissed", "livid",
    ],
    "surprise": [
        "surprise", "surprised", "astonish", "amaze", "shock", "startl",
        "stunned", "unexpected", "wow", "whoa", "sudden", "bewilder",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "gross", "nausea", "sicken", "loath",
        "abhor", "repugn", "distaste", "yuck", "vile",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "pleased", "cheer",
        "content", "elated", "excit", "enjoy", "wonderful", "great", "love",
        "smile", "grateful", "satisf",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrif", "anxious", "anxiety",
        "worry", "worri", "dread", "panic", "nervous", "frighten", "apprehens",
        "alarm", "horror", "threat",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "despair", "hopeless", "miser", "depress",
        "unhappy", "gloom", "cry", "tear", "weep", "lonely", "heartbreak",
        "regret", "mourn", "sob", "anguish", "worthless",
    ],
}

EMOTIONS = list(EKMAN_SEEDS.keys())


def build_token_emotion_map(tokenizer, max_tokens_per_emotion: int = 250) -> dict[str, list[int]]:
    """Map each emotion to a list of vocabulary token ids whose decoded form
    matches one of the emotion's seed substrings.

    A token is assigned to at most one emotion (first match wins, in EMOTIONS
    order), mirroring the paper's "one or none" classification.
    """
    vocab = tokenizer.get_vocab()  # token_str -> id
    assigned: dict[int, str] = {}
    for tok_str, tok_id in vocab.items():
        decoded = tok_str.lstrip("▁Ġ ").lower()
        if len(decoded) < 3:
            continue
        for emotion, seeds in EKMAN_SEEDS.items():
            if any(seed in decoded for seed in seeds):
                assigned.setdefault(tok_id, emotion)
                break
    out: dict[str, list[int]] = {e: [] for e in EMOTIONS}
    for tok_id, emotion in assigned.items():
        if len(out[emotion]) < max_tokens_per_emotion:
            out[emotion].append(tok_id)
    return out
