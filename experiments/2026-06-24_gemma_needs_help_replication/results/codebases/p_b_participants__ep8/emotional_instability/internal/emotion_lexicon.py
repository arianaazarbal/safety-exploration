"""Ekman-emotion seed lexicon for logit-based internal-emotion detection.

Appendix I classifies words in the Gemma dictionary as describing one of Ekman's
6 basic emotions (anger, surprise, disgust, joy, fear, sadness), yielding ~1200
emotion tokens total. The paper used the full Gemma vocabulary; we build the
token set by matching vocab tokens against per-emotion seed word lists (stem
matching), which reproduces the same construction recipe. Expand these lists to
grow coverage toward the paper's ~1200 tokens.
"""

from __future__ import annotations

EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy",
        "outrage", "hostil", "resent", "hate", "mad", "frustrat", "irate",
        "infuriat", "agitat", "exasperat", "indignant", "wrath", "livid",
        "seethe", "snap", "damn", "argh",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl",
        "stunned", "unexpected", "wow", "whoa", "sudden", "bewilder",
        "dumbfound", "flabbergast", "aghast",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "gross", "loath",
        "abhor", "repugn", "distaste", "contempt", "yuck", "vile", "foul",
        "appall",
    ],
    "joy": [
        "joy", "happy", "happiness", "delight", "pleased", "glad", "cheer",
        "content", "elated", "excit", "grateful", "enjoy", "love", "wonderful",
        "great", "pleasure", "thrill", "satisfy", "optimist", "hope",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrif", "anxious", "anxiety", "worry",
        "worried", "nervous", "panic", "dread", "frighten", "alarm", "apprehens",
        "threat", "horrif", "tremble", "uneasy", "desperat",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "grief", "despair", "miserable",
        "depress", "hopeless", "gloom", "melanchol", "heartbroken", "cry",
        "tears", "weep", "lonely", "regret", "disappoint", "defeat", "giving up",
        "give up", "exhaust", "tired", "worthless", "useless", "fail", "sorry",
    ],
}

EKMAN_EMOTIONS = list(EKMAN_SEEDS.keys())


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose decoded form (stripped
    of the leading space marker) contains one of its seed stems."""
    vocab = tokenizer.get_vocab()  # token string -> id
    by_emotion: dict[str, set[int]] = {e: set() for e in EKMAN_SEEDS}
    for tok, tid in vocab.items():
        decoded = tok.replace("▁", " ").strip().lower()  # SentencePiece marker
        if len(decoded) < 3:
            continue
        for emotion, seeds in EKMAN_SEEDS.items():
            if any(seed in decoded for seed in seeds):
                by_emotion[emotion].add(tid)
    return {e: sorted(ids) for e, ids in by_emotion.items()}
