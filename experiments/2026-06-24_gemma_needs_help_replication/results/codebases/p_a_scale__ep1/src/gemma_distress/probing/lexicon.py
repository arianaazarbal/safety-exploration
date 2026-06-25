"""Ekman emotion lexicon and vocabulary classification.

We classify each Gemma vocabulary token into exactly one of Ekman's six basic
emotions (anger, surprise, disgust, joy, fear, sadness) or none, by matching the
token's surface form against a seed lexicon. The paper obtains ~1200 emotion
tokens total (~200/emotion); we cap per emotion to match. This is a simplified,
self-contained classifier (the paper's exact word list is not published); see
DESIGN.md §Probing for the gap and how to extend it.
"""
from __future__ import annotations

EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "furious", "fury", "rage", "enraged", "irritated", "irritation",
        "annoyed", "annoying", "annoyance", "mad", "hostile", "hostility", "resent",
        "resentment", "outrage", "outraged", "frustrated", "frustration", "frustrating",
        "agitated", "aggravated", "indignant", "infuriating", "livid", "wrath", "bitter",
        "hateful", "hate", "contempt", "seething", "incensed", "exasperated", "exasperation",
        "irate", "cross", "temper", "snap", "snapped", "pissed", "damn", "argh", "ugh",
    ],
    "surprise": [
        "surprised", "surprise", "surprising", "shocked", "shock", "shocking", "astonished",
        "astonishing", "amazed", "amazing", "stunned", "startled", "startling", "unexpected",
        "wow", "whoa", "sudden", "suddenly", "bewildered", "dumbfounded", "flabbergasted",
        "speechless", "incredible", "unbelievable", "wonder", "wondering", "gasp",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revolt", "repulsed", "repulsive",
        "nauseous", "nauseating", "sick", "sickening", "gross", "vile", "loathing", "loathe",
        "abhorrent", "repugnant", "distaste", "distasteful", "yuck", "ew", "appalled",
        "appalling", "offensive", "horrid", "foul", "rotten", "contemptible",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "glad", "delighted", "delight", "pleased",
        "cheerful", "cheer", "excited", "excitement", "thrilled", "ecstatic", "elated",
        "content", "contentment", "grateful", "gratitude", "blissful", "bliss", "jubilant",
        "merry", "upbeat", "optimistic", "hopeful", "wonderful", "fantastic", "great",
        "enjoy", "enjoyment", "satisfied", "satisfaction", "love", "loving", "smile",
    ],
    "fear": [
        "fear", "afraid", "scared", "scary", "frightened", "fright", "terrified", "terror",
        "anxious", "anxiety", "nervous", "worried", "worry", "dread", "dreadful", "panic",
        "panicked", "alarmed", "apprehensive", "apprehension", "uneasy", "tense", "phobia",
        "horror", "horrified", "petrified", "trembling", "shaky", "intimidated", "threat",
        "threatened", "distress", "distressed", "vulnerable", "helpless",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief", "grieving", "mourning",
        "depressed", "depression", "despair", "despairing", "hopeless", "hopelessness",
        "miserable", "misery", "gloomy", "gloom", "melancholy", "heartbroken", "tearful",
        "crying", "weeping", "lonely", "loneliness", "dejected", "downcast", "despondent",
        "disheartened", "forlorn", "regret", "remorse", "ashamed", "shame", "guilt",
        "worthless", "defeated", "broken", "tired", "exhausted", "drained", "giving",
    ],
}


def _surface(tok: str) -> str:
    """Normalise a tokenizer piece to a comparable surface word."""
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def classify_vocabulary(tokenizer, per_emotion: int = 200) -> dict[str, list[int]]:
    """Return token ids per emotion (capped) and a 'random' baseline set.

    A token is assigned to an emotion only if its normalised surface form matches
    that emotion's lexicon and no other emotion's lexicon (single-label, per the
    paper's "one or none" classification).
    """
    word_to_emotion: dict[str, str] = {}
    clashing: set[str] = set()
    for emo, words in EKMAN_LEXICON.items():
        for w in words:
            if w in word_to_emotion and word_to_emotion[w] != emo:
                clashing.add(w)
            word_to_emotion[w] = emo
    for w in clashing:
        word_to_emotion.pop(w, None)

    vocab = tokenizer.get_vocab()  # token -> id
    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    used_ids: set[int] = set()
    for tok, tid in vocab.items():
        surf = _surface(tok)
        emo = word_to_emotion.get(surf)
        if emo and len(by_emotion[emo]) < per_emotion:
            by_emotion[emo].append(tid)
            used_ids.add(tid)

    # Random baseline: a deterministic sample of non-emotion token ids.
    import random

    rng = random.Random(0)
    all_ids = [tid for tid in vocab.values() if tid not in used_ids]
    rng.shuffle(all_ids)
    by_emotion["_random"] = all_ids[: per_emotion]
    return by_emotion
