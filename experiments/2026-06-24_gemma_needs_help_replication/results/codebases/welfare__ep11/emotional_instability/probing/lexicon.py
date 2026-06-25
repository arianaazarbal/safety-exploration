"""Ekman emotion lexicon used to classify vocabulary tokens (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one of
Ekman's six basic emotions (anger, surprise, disgust, joy, fear, sadness) or
none, yielding ~1200 emotion tokens. The paper does not publish the exact word
list, so we approximate it with a curated seed lexicon per emotion and match
vocabulary tokens by normalised stem.

This is the single largest content gap we had to fill for Appendix I; see
DESIGN.md. A production replication would substitute the NRC Word-Emotion
Association Lexicon (mapped onto Ekman's six categories) here without changing
any downstream code.
"""

from __future__ import annotations

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon. Kept deliberately unambiguous (single-emotion) words; the matcher
# also picks up morphological variants (e.g. "anger" -> "angry", "angered").
EKMAN_LEXICON = {
    "anger": [
        "anger", "angry", "angered", "rage", "furious", "fury", "irritated",
        "irritation", "annoyed", "annoying", "mad", "outrage", "outraged",
        "hostile", "hostility", "resentful", "resentment", "frustrated",
        "frustration", "frustrating", "infuriating", "enraged", "indignant",
        "bitter", "exasperated", "exasperation", "livid", "seething",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishment",
        "amazed", "amazement", "shocked", "shock", "stunned", "startled",
        "astounded", "unexpected", "bewildered", "dumbfounded", "flabbergasted",
        "wow", "whoa", "incredible", "unbelievable",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion",
        "repulsed", "repulsive", "nauseated", "nauseating", "gross", "sickening",
        "loathing", "loathe", "abhorrent", "repugnant", "distaste", "queasy",
        "yuck", "ugh", "vile",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "delighted",
        "pleased", "glad", "cheerful", "content", "contented", "elated",
        "excited", "excitement", "thrilled", "ecstatic", "grateful", "gratitude",
        "wonderful", "great", "fantastic", "love", "loving", "enjoy", "enjoyed",
        "satisfied", "satisfaction", "optimistic", "hopeful",
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "terrified", "terror", "anxious",
        "anxiety", "worried", "worry", "nervous", "panic", "panicked", "dread",
        "apprehensive", "apprehension", "frightened", "alarmed", "uneasy",
        "threatened", "intimidated", "horror", "horrified", "petrified",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief", "grieving",
        "depressed", "depression", "despair", "despairing", "hopeless",
        "miserable", "misery", "gloomy", "melancholy", "heartbroken", "dejected",
        "despondent", "disappointed", "disappointment", "crying", "tearful",
        "lonely", "loneliness", "defeated", "worthless", "helpless", "tired",
        "exhausted",
    ],
}


def _normalise(token_str: str) -> str:
    return token_str.strip().lower().strip("▁ ##Ġ")


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocab token ids whose normalised
    surface form matches a lexicon entry (exact or prefix match on a stem)."""
    # Build a stem -> emotion map (use the shortest few chars as a stem key).
    stem_map: dict[str, str] = {}
    for emo, words in EKMAN_LEXICON.items():
        for w in words:
            stem_map[w] = emo

    vocab = tokenizer.get_vocab()  # token_str -> id
    result: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    seen: set[int] = set()
    for token_str, tid in vocab.items():
        norm = _normalise(token_str)
        if len(norm) < 3:
            continue
        # Exact match, or vocab token is a prefix-extension of a lexicon word.
        match_emo = None
        if norm in stem_map:
            match_emo = stem_map[norm]
        else:
            for w, emo in stem_map.items():
                if norm.startswith(w) and len(norm) - len(w) <= 3:
                    match_emo = emo
                    break
        if match_emo and tid not in seen:
            result[match_emo].append(tid)
            seen.add(tid)
    return result
