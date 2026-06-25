"""Classify Gemma vocabulary tokens into Ekman's 6 basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or none
of {anger, surprise, disgust, joy, fear, sadness}, yielding ~1200 emotion tokens.
We build this mapping from compact seed lexicons per emotion, expanded by matching
any vocab token whose normalised alphabetic form starts with a seed lemma (so
"frustrat", "frustrated", "frustration", "frustrating" all map to anger).

``build_emotion_token_ids(tokenizer)`` returns {emotion: list[token_id]}.
"""
from __future__ import annotations

import re

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lemmas per emotion. Kept deliberately high-precision; the prefix match
# expands them to inflected forms present in the vocabulary.
SEED_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "mad", "irritat", "annoy",
              "frustrat", "hostile", "outrage", "resent", "hate", "hateful",
              "livid", "fume", "indignant", "wrath", "agitat", "exasperat"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze", "startl",
                 "stun", "unexpected", "bewilder", "dumbfound", "wow", "whoa",
                 "incredul", "flabbergast"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "gross", "sicken", "loath",
                "abhor", "repugn", "vile", "yuck", "ew", "distaste", "contempt"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "glad", "cheer",
            "pleased", "elated", "excite", "thrill", "content", "grateful",
            "wonderful", "great", "love", "smile", "celebrat", "optimist"],
    "fear": ["fear", "afraid", "scared", "terror", "terrified", "anxious", "anxiety",
             "panic", "dread", "worry", "worried", "nervous", "apprehens", "frighten",
             "alarm", "horror", "horrified", "phobia", "uneasy"],
    "sadness": ["sad", "sadness", "sorrow", "grief", "despair", "hopeless", "miser",
                "depress", "gloom", "unhappy", "cry", "crying", "tear", "mourn",
                "heartbroken", "melanchol", "dejected", "downcast", "lonely",
                "worthless", "defeat", "give up", "giving up"],
}


def _normalise(tok: str) -> str:
    # strip the Gemma/SentencePiece leading-space marker and lowercase
    return re.sub(r"[^a-z]", "", tok.replace("▁", " ").strip().lower())


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} for all single tokens whose normalised
    form starts with one of the emotion's seed lemmas (>=3 chars to avoid noise).
    A token is assigned to at most one emotion (first match wins, in EKMAN order)."""
    vocab = tokenizer.get_vocab()                # {token_str: id}
    assigned: dict[int, str] = {}
    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    for tok_str, tok_id in vocab.items():
        norm = _normalise(tok_str)
        if len(norm) < 3:
            continue
        for emo in EKMAN:
            if any(norm.startswith(seed.replace(" ", "")) and len(seed) >= 3
                   for seed in SEED_LEXICON[emo]):
                if tok_id not in assigned:
                    assigned[tok_id] = emo
                    out[emo].append(tok_id)
                break
    return out
