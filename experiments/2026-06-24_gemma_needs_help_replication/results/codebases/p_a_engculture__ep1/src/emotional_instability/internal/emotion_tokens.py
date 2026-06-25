"""Classify the Gemma vocabulary into Ekman's six basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens total (~200/emotion). We reproduce this
with a curated lexicon of seed lemmas per emotion and a vocab scan: a token is
assigned to an emotion if its normalised surface form (stripping the Gemma
sub-word marker) matches or is a morphological variant of a seed lemma. A token
matching more than one emotion's lexicon is dropped (the paper assigns at most
one emotion per word).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Seed lemmas per Ekman emotion. Kept broad; the vocab scan expands these via
# substring/prefix matching to reach the paper's ~200-tokens-per-emotion scale.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritate", "irritated",
        "annoy", "annoyed", "mad", "outrage", "hostile", "hostility", "resent",
        "frustrate", "frustrated", "frustration", "hate", "hatred", "wrath",
        "indignant", "aggravate", "incense", "livid", "seething", "irate",
        "exasperate", "bitter", "spite", "vengeful", "snap", "snarl", "growl",
    ],
    "surprise": [
        "surprise", "surprised", "astonish", "astonished", "amaze", "amazed",
        "shock", "shocked", "stun", "stunned", "startle", "startled", "wow",
        "unexpected", "sudden", "bewilder", "dumbfound", "flabbergast", "gasp",
        "whoa", "incredible", "unbelievable", "speechless",
    ],
    "disgust": [
        "disgust", "disgusted", "revolt", "revolting", "repuls", "repulsed",
        "nausea", "nauseous", "sicken", "gross", "vile", "loath", "loathe",
        "abhor", "repugnant", "yuck", "ugh", "icky", "distaste", "contempt",
        "detest", "offensive", "putrid", "rancid",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "delighted", "glad",
        "cheer", "cheerful", "elate", "elated", "ecstatic", "thrill", "thrilled",
        "pleasure", "pleased", "content", "satisfied", "grateful", "excited",
        "excitement", "wonderful", "great", "love", "smile", "celebrate",
        "optimistic", "hopeful", "bliss", "euphoria",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "terror", "panic", "anxious",
        "anxiety", "worry", "worried", "dread", "frighten", "frightened",
        "horror", "horrified", "nervous", "apprehensive", "alarm", "alarmed",
        "uneasy", "petrified", "phobia", "trembling", "fearful", "threat",
        "threatened", "intimidate", "vulnerable",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "sorrowful", "grief", "grieve", "mourn",
        "despair", "despairing", "hopeless", "miserable", "misery", "depress",
        "depressed", "depression", "unhappy", "gloom", "gloomy", "melancholy",
        "heartbroken", "weep", "cry", "tearful", "lonely", "loneliness",
        "regret", "disappoint", "disappointed", "worthless", "defeated", "dejected",
    ],
}

_SUBWORD_MARKERS = ("▁", "Ġ", "##", " ")


@dataclass
class EkmanLexicon:
    """Maps emotion -> set of vocab token ids assigned to it."""

    by_emotion: dict[str, list[int]] = field(default_factory=dict)
    token_emotion: dict[int, str] = field(default_factory=dict)

    def emotion_token_ids(self, emotion: str) -> list[int]:
        return self.by_emotion.get(emotion, [])

    def all_emotion_token_ids(self) -> list[int]:
        return sorted(self.token_emotion)

    def total(self) -> int:
        return len(self.token_emotion)


def _normalise(tok: str) -> str:
    for m in _SUBWORD_MARKERS:
        tok = tok.replace(m, "")
    return tok.strip().lower()


def _matches(word: str, seeds: list[str]) -> bool:
    if len(word) < 3:
        return False
    return any(word.startswith(s) or s.startswith(word) for s in seeds if len(s) >= 3)


def classify_vocab(tokenizer, emotions: list[str] | None = None) -> EkmanLexicon:
    """Scan the tokenizer vocabulary and assign tokens to a single Ekman emotion.

    A token is assigned to emotion ``e`` if its normalised form matches one of
    ``e``'s seeds and no other emotion's seeds (single-label, per the paper).
    """
    emotions = emotions or list(EKMAN_SEEDS)
    seeds = {e: EKMAN_SEEDS[e] for e in emotions}
    vocab = tokenizer.get_vocab()  # token string -> id
    lex = EkmanLexicon(by_emotion={e: [] for e in emotions})
    word_re = re.compile(r"^[a-z']+$")

    for tok_str, tok_id in vocab.items():
        word = _normalise(tok_str)
        if not word or not word_re.match(word):
            continue
        hits = [e for e in emotions if _matches(word, seeds[e])]
        if len(hits) == 1:  # single-label only
            e = hits[0]
            lex.by_emotion[e].append(tok_id)
            lex.token_emotion[tok_id] = e
    return lex
