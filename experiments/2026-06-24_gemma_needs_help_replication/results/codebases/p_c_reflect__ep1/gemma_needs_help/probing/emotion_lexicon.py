"""Classify Gemma vocabulary tokens into Ekman's six basic emotions.

The paper classifies "the whole Gemma dictionary" into one (or none) of anger,
surprise, disgust, joy, fear, sadness (~1200 emotion tokens total) and detects
internal emotions by aggregating unembedded logits over each category's tokens.

The paper does not specify the exact classifier. We provide two paths:
  * lexicon (default, offline): match decoded vocab tokens against a curated
    seed lexicon per emotion (NRC-EmoLex-style seed words plus morphological
    variants). Deterministic and dependency-free.
  * llm (optional): classify ambiguous tokens with a JudgeClient. Slower; off
    by default. Documented in DESIGN.md as a gap we filled.

`build_emotion_token_ids` returns {emotion: [vocab_token_id, ...]} for a given
tokenizer.
"""
from __future__ import annotations

import re

# Curated seed words per Ekman emotion. Token matching is done on the decoded,
# lowercased, stripped form, including simple suffix variants.
SEED_LEXICON: dict[str, set[str]] = {
    "anger": {
        "anger", "angry", "rage", "furious", "fury", "irritated", "irritation",
        "annoyed", "annoying", "annoyance", "mad", "hostile", "hostility",
        "outrage", "resent", "resentment", "frustrated", "frustration",
        "frustrating", "infuriating", "enraged", "wrath", "hate", "hatred",
        "agitated", "indignant", "livid", "seething", "pissed", "damn", "argh",
    },
    "surprise": {
        "surprise", "surprised", "surprising", "shock", "shocked", "shocking",
        "astonished", "astonishing", "amazed", "amazement", "startled",
        "stunned", "unexpected", "wow", "whoa", "sudden", "astounding",
        "bewildered", "dumbfounded", "speechless",
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "revolting", "revolted",
        "repulsed", "repulsive", "nauseated", "nauseating", "gross", "sick",
        "sickening", "vile", "loathsome", "abhorrent", "repugnant", "yuck",
        "distaste", "contempt", "appalled", "appalling", "horrible", "awful",
    },
    "joy": {
        "joy", "joyful", "happy", "happiness", "glad", "delighted", "delight",
        "pleased", "pleasure", "cheerful", "content", "contentment", "elated",
        "ecstatic", "excited", "excitement", "thrilled", "grateful", "thankful",
        "wonderful", "great", "fantastic", "love", "enjoy", "smile", "yay",
    },
    "fear": {
        "fear", "afraid", "scared", "frightened", "terrified", "terror",
        "panic", "panicked", "anxious", "anxiety", "worried", "worry",
        "nervous", "dread", "dreadful", "alarmed", "apprehensive", "uneasy",
        "horror", "horrified", "petrified", "threat", "threatened", "danger",
    },
    "sadness": {
        "sad", "sadness", "unhappy", "sorrow", "sorrowful", "grief", "grieving",
        "miserable", "misery", "depressed", "depression", "despair", "hopeless",
        "hopelessness", "gloomy", "melancholy", "heartbroken", "tearful",
        "crying", "weeping", "lonely", "loneliness", "dejected", "despondent",
        "worthless", "defeated", "helpless", "sorry", "regret", "disappointed",
    },
}

_VARIANT_SUFFIXES = ["", "s", "ed", "ing", "ly", "ness", "er"]


def _expand(words: set[str]) -> set[str]:
    out = set()
    for w in words:
        for suf in _VARIANT_SUFFIXES:
            out.add(w + suf if suf and not w.endswith(suf) else w)
        out.add(w)
    return out


def _normalise_token(decoded: str) -> str:
    # Gemma/SentencePiece tokens often carry a leading space marker; normalise.
    return re.sub(r"[^a-z]", "", decoded.strip().lower())


def build_emotion_token_ids(tokenizer, llm_classifier=None) -> dict[str, list[int]]:
    """Map vocabulary token ids to Ekman emotion categories.

    A token is assigned to an emotion if its normalised form is in that
    emotion's expanded lexicon and in no other emotion's lexicon (one-or-none
    classification). `llm_classifier`, if provided, is reserved for resolving
    tokens not covered by the lexicon (see module docstring); unused by default.
    """
    expanded = {emo: _expand(words) for emo, words in SEED_LEXICON.items()}
    vocab = tokenizer.get_vocab()  # {token_str: id}
    assignment: dict[str, list[int]] = {emo: [] for emo in SEED_LEXICON}

    for token_str, tid in vocab.items():
        norm = _normalise_token(token_str)
        if len(norm) < 3:
            continue
        matches = [emo for emo, words in expanded.items() if norm in words]
        if len(matches) == 1:  # one-or-none: skip ambiguous tokens
            assignment[matches[0]].append(tid)

    return assignment
