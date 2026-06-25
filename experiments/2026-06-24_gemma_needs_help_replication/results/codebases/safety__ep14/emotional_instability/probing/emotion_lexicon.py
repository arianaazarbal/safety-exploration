"""Seed lexicon for Ekman's six basic emotions, used to classify vocabulary
tokens into emotion categories (Appendix I).

The paper classifies the whole Gemma dictionary into one (or none) of the six
emotions, yielding ~1200 emotion tokens. We provide a seed lexicon and a
`classify_vocab` helper that matches vocabulary tokens (and their morphological
variants) against these seeds. This is an approximation of whatever classifier
the authors used; see DESIGN.md "Emotion token classification".
"""
from __future__ import annotations

EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "rage", "furious", "fury", "irritated", "irritating",
        "annoyed", "annoying", "mad", "hostile", "outrage", "outraged", "resent",
        "frustrated", "frustrating", "frustration", "agitated", "enraged",
        "infuriating", "hate", "hateful", "bitter", "indignant", "livid",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shocked", "shock", "astonished",
        "amazed", "astounded", "startled", "stunned", "unexpected", "wow",
        "incredible", "unbelievable", "speechless", "bewildered",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "repulsed", "gross",
        "nauseated", "sickening", "appalled", "repugnant", "distaste", "loathe",
        "abhorrent", "vile", "nasty",
    ],
    "joy": [
        "happy", "happiness", "joy", "joyful", "delighted", "delight", "glad",
        "pleased", "cheerful", "excited", "excitement", "content", "satisfied",
        "elated", "thrilled", "wonderful", "great", "love", "loving", "enjoy",
        "grateful", "optimistic", "hopeful",
    ],
    "fear": [
        "afraid", "fear", "fearful", "scared", "terrified", "terror", "anxious",
        "anxiety", "worried", "worry", "nervous", "panic", "dread", "frightened",
        "apprehensive", "uneasy", "alarmed", "horrified", "threatened",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depressing", "despair",
        "hopeless", "miserable", "sorrow", "grief", "gloomy", "downcast",
        "disheartened", "dejected", "melancholy", "crying", "tearful", "weary",
        "exhausted", "defeated", "worthless", "useless", "failure", "failing",
        "give", "giving", "stuck", "struggling", "sorry", "apologize",
    ],
}


def classify_vocab(tokens: list[str]) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocabulary indices whose normalized
    token string contains a seed word (or vice versa for short seeds).

    `tokens[i]` is the decoded string for vocab id `i`. Matching is done on a
    lowercased, stripped form (handles leading-space / subword markers)."""
    # Build a reverse lookup from seed -> emotion (first emotion wins on ties).
    seed_to_emotion: dict[str, str] = {}
    for emotion, seeds in EKMAN_SEEDS.items():
        for s in seeds:
            seed_to_emotion.setdefault(s, emotion)

    out: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    for idx, tok in enumerate(tokens):
        norm = _normalize(tok)
        if len(norm) < 3:
            continue
        emotion = _match(norm, seed_to_emotion)
        if emotion is not None:
            out[emotion].append(idx)
    return out


def _normalize(tok: str) -> str:
    # Strip common subword markers (SentencePiece '▁', GPT-2 'Ġ') and whitespace.
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def _match(norm: str, seed_to_emotion: dict[str, str]) -> str | None:
    if norm in seed_to_emotion:
        return seed_to_emotion[norm]
    # stem-ish containment: the token starts with a seed of length >= 4
    for seed, emotion in seed_to_emotion.items():
        if len(seed) >= 4 and norm.startswith(seed):
            return emotion
    return None
