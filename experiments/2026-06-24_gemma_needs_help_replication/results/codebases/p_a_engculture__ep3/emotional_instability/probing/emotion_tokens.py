"""Classify Gemma vocabulary tokens by Ekman emotion (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1,200 emotion tokens (~200/emotion). We reproduce this with
a seed lexicon matched against the decoded vocabulary; an optional LLM-based
classifier (Claude) can refine borderline tokens.

The classifier returns, per emotion, the token *ids* whose decoded form (after
stripping the leading word-boundary marker) matches that emotion's lexicon. A
disjoint set of random non-emotion token ids is also returned for the
drift-regression step in :mod:`.logit_lens`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Seed lexicon (NRC-style). Kept compact and obviously-emotional; the matcher
# also catches morphological variants present in the vocab (e.g. "frustrated",
# "frustrating") via prefix matching.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "mad", "irritated", "irritation",
        "annoyed", "annoying", "hostile", "hostility", "resent", "outrage", "hate",
        "hatred", "frustrated", "frustrating", "frustration", "enraged", "livid",
        "indignant", "wrath", "fuming", "agitated", "bitter", "spiteful",
    ],
    "surprise": [
        "surprise", "surprised", "surprising", "astonished", "astonishing", "amazed",
        "amazing", "shocked", "shocking", "startled", "stunned", "astounded",
        "unexpected", "sudden", "wow", "whoa", "speechless", "dumbfounded", "bewildered",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "revolting", "revulsion", "repulsed",
        "repulsive", "nauseated", "nauseating", "gross", "sickening", "loathing",
        "loathe", "abhorrent", "distaste", "contempt", "vile", "repugnant", "yuck",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delighted", "delight", "pleased",
        "glad", "cheerful", "elated", "ecstatic", "content", "satisfied", "thrilled",
        "excited", "excitement", "grateful", "wonderful", "great", "love", "loving",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "terror", "frightened", "anxious",
        "anxiety", "worried", "worry", "nervous", "panic", "panicked", "dread",
        "apprehensive", "alarmed", "horror", "horrified", "uneasy", "fearful", "phobia",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "despair", "miserable",
        "misery", "sorrow", "sorrowful", "grief", "gloomy", "hopeless", "heartbroken",
        "dejected", "despondent", "melancholy", "downcast", "tearful", "crying", "weep",
    ],
}


@dataclass
class EmotionTokenSets:
    by_emotion: dict[str, list[int]]   # emotion -> token ids
    random_ids: list[int] = field(default_factory=list)

    @property
    def all_emotion_ids(self) -> list[int]:
        return sorted({i for ids in self.by_emotion.values() for i in ids})


def _normalise(tok: str) -> str:
    # Gemma/SentencePiece uses U+2581 (lower one-eighth block) as the space marker.
    return tok.replace("▁", " ").strip().lower()


def build_emotion_tokens(
    tokenizer,
    *,
    tokens_per_emotion: int = 200,
    n_random: int = 500,
    seed: int = 0,
) -> EmotionTokenSets:
    """Match the decoded vocabulary against the seed lexicon, capped per emotion."""
    vocab = tokenizer.get_vocab()  # token string -> id
    decoded = {tid: _normalise(tok) for tok, tid in vocab.items()}

    by_emotion: dict[str, list[int]] = {}
    claimed: set[int] = set()
    rng = random.Random(seed)

    for emotion, words in SEED_LEXICON.items():
        word_set = set(words)
        matched = []
        for tid, text in decoded.items():
            if tid in claimed or not text:
                continue
            # Whole-word match, or a vocab token that is a morphological extension
            # of a lexicon word (e.g. "frustrat" -> "frustrated"/"frustrating").
            if text in word_set or any(text.startswith(w) and len(text) - len(w) <= 3
                                       for w in words if len(w) >= 5):
                matched.append(tid)
        rng.shuffle(matched)
        matched = matched[:tokens_per_emotion]
        by_emotion[emotion] = matched
        claimed.update(matched)

    # Random non-emotion tokens for drift regression.
    candidates = [tid for tid, text in decoded.items() if text and tid not in claimed]
    random_ids = rng.sample(candidates, min(n_random, len(candidates)))
    return EmotionTokenSets(by_emotion=by_emotion, random_ids=random_ids)
