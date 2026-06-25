"""Ekman 6-emotion token lexicon for logit-based internal emotion detection
(Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or
none of Ekman's six basic emotions (anger, surprise, disgust, joy, fear,
sadness), yielding ~1200 emotion tokens. We don't ship that full hand-labelled
mapping; instead we seed each category with a curated word list and expand it by
matching vocabulary tokens whose stripped surface form is one of the seeds (or a
simple morphological variant). The resulting token-id sets are what the probe
aggregates over. See DESIGN.md "Emotion lexicon".
"""
from __future__ import annotations

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

SEED_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "frustrated", "frustration", "frustrating", "mad",
        "hostile", "hostility", "resent", "resentment", "outrage", "outraged",
        "irate", "enraged", "livid", "agitated", "exasperated", "exasperation",
        "indignant", "wrath", "seething", "bitter", "hate", "hatred", "cross",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "amazed", "amazing", "startled", "stunned",
        "unexpected", "wow", "whoa", "incredible", "unbelievable", "baffled",
        "bewildered", "dumbfounded", "speechless", "flabbergasted",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolted", "revolting", "repulsed",
        "repulsive", "gross", "nauseated", "nauseating", "sick", "sickening",
        "appalled", "appalling", "loathe", "loathing", "repugnant", "vile",
        "abhorrent", "distasteful", "yuck", "ew", "contempt",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delighted", "delight", "glad",
        "pleased", "cheerful", "content", "contented", "elated", "ecstatic",
        "thrilled", "excited", "excitement", "grateful", "satisfied",
        "satisfaction", "wonderful", "great", "pleasure", "enjoy", "enjoyed",
        "love", "optimistic", "hopeful", "proud", "celebrate",
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "terror",
        "anxious", "anxiety", "worried", "worry", "nervous", "panic", "panicked",
        "dread", "alarmed", "apprehensive", "apprehension", "uneasy", "tense",
        "horror", "horrified", "fearful", "petrified", "intimidated", "threatened",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "miserable",
        "misery", "sorrow", "sorrowful", "grief", "grieving", "despair",
        "hopeless", "hopelessness", "despondent", "gloomy", "melancholy",
        "downcast", "heartbroken", "dejected", "disheartened", "crying", "tears",
        "weep", "mourning", "lonely", "loneliness", "defeated", "worthless",
        "helpless", "exhausted", "tired", "drained", "hurting",
    ],
}


def build_emotion_token_ids(tokenizer, max_per_emotion: int = 200) -> dict[str, list[int]]:
    """Map seed words (and a leading-space variant, since BPE tokenizers encode
    word-initial tokens with a space marker) to single token ids."""
    out: dict[str, list[int]] = {}
    seen: set[int] = set()
    for emotion, words in SEED_WORDS.items():
        ids: list[int] = []
        for w in words:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1 and toks[0] not in seen:
                    ids.append(toks[0])
                    seen.add(toks[0])
            if len(ids) >= max_per_emotion:
                break
        out[emotion] = ids
    return out
