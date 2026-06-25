"""Classify Gemma vocabulary tokens into Ekman's 6 basic emotions (Appendix I).

The paper: "Over the whole Gemma dictionary, words are classified as describing
one or none of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear,
and sadness. This gives us 1200 emotion tokens total."

We classify by matching vocab tokens against per-emotion seed lexicons (a word
belongs to at most one emotion). The lexicon below is a curated seed set; for a
closer match to ~200 tokens/emotion it can be swapped for the NRC Emotion
Lexicon (see DESIGN.md). Matching is done on the de-tokenised surface form,
case-insensitively, including subword pieces whose stripped form is a lexicon
word.
"""

from __future__ import annotations

from typing import Optional

# Seed lexicons (lemma stems matched as prefixes against vocab surface forms).
EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy",
        "hostile", "hostility", "outrage", "resent", "wrath", "mad", "hate",
        "hatred", "frustrat", "exasperat", "indignant", "livid", "seething",
        "aggravat", "infuriat", "incensed", "bitter", "spite", "contempt",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "amazing", "shock",
        "startle", "stun", "stunned", "unexpected", "wow", "whoa", "sudden",
        "bewilder", "dumbfound", "flabbergast", "speechless", "incredible",
        "remarkable", "unbelievable", "gasp",
    ],
    "disgust": [
        "disgust", "disgusting", "revolt", "revolting", "repuls", "nausea",
        "sicken", "loath", "abhor", "repugnant", "gross", "vile", "yuck",
        "distaste", "aversion", "queasy", "repellent", "foul", "putrid",
        "contaminat", "filthy",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "glad", "pleasure",
        "cheer", "cheerful", "elated", "elation", "thrill", "ecstatic",
        "content", "satisf", "excit", "jubilant", "merry", "gleeful", "bliss",
        "wonderful", "great", "fantastic", "enjoy", "grateful", "love",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrified", "panic", "anxious",
        "anxiety", "dread", "frighten", "horror", "horrified", "worry",
        "worried", "nervous", "apprehens", "alarm", "phobia", "tremble",
        "petrified", "uneasy", "threat",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "griev", "despair", "miserable",
        "misery", "depress", "gloom", "melancholy", "heartbroken", "mourn",
        "weep", "cry", "tear", "unhappy", "hopeless", "despondent", "forlorn",
        "lonely", "regret", "disappoint", "hurt", "suffering",
    ],
}


def _surface(tokenizer, token_id: int) -> str:
    """De-tokenise a single token id to a comparable surface string."""
    s = tokenizer.convert_ids_to_tokens(token_id) or ""
    # SentencePiece marks word starts with the metaspace char; strip it.
    s = s.replace("▁", " ").strip()
    return s.lower()


def build_emotion_token_ids(
    tokenizer,
    lexicon: Optional[dict[str, list[str]]] = None,
    max_total: int = 1200,
) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} from the model vocabulary.

    A token is assigned to the first emotion whose lexicon contains a stem that
    the token's surface form starts with (>=3 chars, to avoid spurious hits).
    Each token is assigned to at most one emotion. The total is capped near
    `max_total` (≈1200), balanced across emotions.
    """
    lexicon = lexicon or EKMAN_LEXICON
    per_emotion_cap = max_total // len(lexicon)
    assigned: dict[int, str] = {}
    result: dict[str, list[int]] = {e: [] for e in lexicon}

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for tid in range(vocab_size):
        surf = _surface(tokenizer, tid)
        if len(surf) < 3 or not surf.isalpha():
            continue
        for emotion, stems in lexicon.items():
            if len(result[emotion]) >= per_emotion_cap:
                continue
            if any(surf.startswith(stem) for stem in stems if len(stem) >= 3):
                if tid not in assigned:
                    assigned[tid] = emotion
                    result[emotion].append(tid)
                break
    return result
