"""Build the Ekman emotion-token dictionary over Gemma's vocabulary (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or
none of Ekman's six basic emotions (anger, surprise, disgust, joy, fear,
sadness), yielding ~1200 emotion tokens (~200/emotion). We reproduce this with a
seed lexicon per emotion plus morphological/substring matching against decoded
vocabulary tokens. An optional LLM-classification path (``classify_with_llm``)
mirrors the paper's "classified ... as describing one or none" wording more
closely but needs an API key and is slower.
"""
from __future__ import annotations

import json
from pathlib import Path

# Seed lexicons. Matching is done on decoded, lowercased, stripped tokens.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
              "annoyed", "annoying", "hostile", "hostility", "mad", "outrage",
              "resentment", "frustrated", "frustration", "indignant", "wrath",
              "enraged", "agitated", "exasperated", "incensed", "livid", "irate"],
    "surprise": ["surprise", "surprised", "surprising", "shock", "shocked", "shocking",
                 "astonished", "astonishing", "amazed", "amazement", "startled",
                 "stunned", "unexpected", "bewildered", "dumbfounded", "astounded"],
    "disgust": ["disgust", "disgusted", "disgusting", "revolting", "revulsion",
                "repulsed", "repulsive", "nauseated", "gross", "sickening", "loathing",
                "contempt", "distaste", "abhorrent", "repugnant", "vile"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "delighted", "pleased",
            "glad", "cheerful", "content", "elated", "thrilled", "excited", "ecstatic",
            "grateful", "satisfied", "optimistic", "hopeful", "proud", "enthusiastic"],
    "fear": ["fear", "afraid", "scared", "terrified", "terror", "frightened", "anxious",
             "anxiety", "worried", "worry", "dread", "panic", "nervous", "apprehensive",
             "alarmed", "fearful", "threatened", "uneasy", "horror", "petrified"],
    "sadness": ["sad", "sadness", "unhappy", "miserable", "depressed", "depression",
                "despair", "hopeless", "grief", "sorrow", "sorrowful", "gloomy",
                "melancholy", "dejected", "despondent", "heartbroken", "mournful",
                "downcast", "forlorn", "woeful", "crying", "tears"],
}


def build_emotion_token_ids(tokenizer, *, max_per_emotion: int = 200) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} by matching vocab tokens to seed lexicons.

    A vocab token matches an emotion if its decoded, normalised form contains (or
    is contained by) a seed word. Tokens matching more than one emotion are
    dropped (the paper assigns each word to one or none).
    """
    vocab = tokenizer.get_vocab()  # token_str -> id
    assigned: dict[int, str] = {}
    conflicts: set[int] = set()

    for tok_str, tok_id in vocab.items():
        norm = _normalise_token(tok_str)
        if len(norm) < 3:
            continue
        match = _match_emotion(norm)
        if match is None:
            continue
        if tok_id in assigned and assigned[tok_id] != match:
            conflicts.add(tok_id)
        else:
            assigned[tok_id] = match

    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    for tok_id, emotion in assigned.items():
        if tok_id in conflicts:
            continue
        if len(by_emotion[emotion]) < max_per_emotion:
            by_emotion[emotion].append(tok_id)
    return by_emotion


def _normalise_token(tok_str: str) -> str:
    # Strip SentencePiece/BPE leading-space markers and lowercase.
    return tok_str.replace("▁", "").replace("Ġ", "").strip().lower()


def _match_emotion(norm: str) -> str | None:
    matches = set()
    for emotion, seeds in EKMAN_SEEDS.items():
        for seed in seeds:
            if norm == seed or norm.startswith(seed) or seed.startswith(norm):
                matches.add(emotion)
                break
    if len(matches) == 1:
        return next(iter(matches))
    return None  # 0 or >1 matches -> unassigned (one-or-none rule)


def save(by_emotion: dict[str, list[int]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(by_emotion, f)
