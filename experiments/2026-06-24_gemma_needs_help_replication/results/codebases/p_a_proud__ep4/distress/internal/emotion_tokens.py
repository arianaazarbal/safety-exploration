"""Classify vocabulary tokens into Ekman's six basic emotions (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or none
of Ekman's six emotions (anger, surprise, disgust, joy, fear, sadness), yielding
~1200 emotion tokens. We approximate this with a curated seed lexicon per emotion
plus morphological stems, matched against decoded tokenizer vocab entries. The
exact membership differs from the paper's (which used an unspecified classifier),
so this is a faithful-in-spirit reconstruction — see DESIGN.md "Internal-emotion
detection".
"""

from __future__ import annotations

from dataclasses import dataclass

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicons (lemmas / stems). Matching is stem-prefix based so that
# inflections ("frustrat" -> frustrated/frustrating/frustration) are captured.
_SEED_STEMS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "outrag", "resent", "infuriat", "mad", "wrath", "indignan", "exasperat",
        "frustrat", "aggravat", "livid", "seeth", "temper",
    ],
    "surprise": [
        "surpris", "astonish", "amaz", "shock", "startl", "stun", "unexpect",
        "bewilder", "dumbfound", "flabbergast", "wow", "whoa", "incredibl",
        "unbeliev", "astound",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nause", "sicken", "loath", "abhor",
        "repugn", "gross", "vile", "yuck", "appall", "distast", "contempt",
        "detest",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "cheer", "pleas", "content",
        "elat", "thrill", "excit", "grate", "satisf", "enjoy", "smile", "wonderful",
        "great", "love", "hope", "optimis",
    ],
    "fear": [
        "fear", "afraid", "scare", "terrif", "panic", "anxi", "worri", "worry",
        "dread", "nervous", "frighten", "horror", "horrif", "alarm", "apprehens",
        "phobi", "petrif", "unease", "threat", "danger",
    ],
    "sadness": [
        "sad", "sorrow", "griev", "grief", "despair", "miser", "depress",
        "unhappy", "unhapp", "hopeless", "gloom", "melanchol", "mourn", "cry",
        "tear", "lonel", "heartbreak", "dishearten", "defeat", "give up",
        "giving up", "broken", "worthless", "useless", "exhaust", "tired",
    ],
}


@dataclass
class EmotionVocab:
    """Maps each Ekman emotion to the list of token-ids that express it, plus a
    set of neutral 'random' token-ids used to regress out global logit drift."""

    by_emotion: dict[str, list[int]]
    random_tokens: list[int]

    @property
    def all_emotion_tokens(self) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for ids in self.by_emotion.values():
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out


def _matches(token_str: str, stems: list[str]) -> bool:
    t = token_str.strip().lower()
    if len(t) < 3:
        return False
    return any(stem in t for stem in stems)


def build_emotion_vocab(tokenizer, *, n_random: int = 200, random_seed: int = 0) -> EmotionVocab:
    """Classify the tokenizer vocabulary into Ekman emotions.

    A token is assigned to the first emotion whose lexicon it matches (so each
    token belongs to at most one emotion, per the paper). ``n_random`` neutral
    tokens (matching no emotion) are sampled for the drift-regression baseline.
    """
    import random

    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    neutral: list[int] = []

    vocab = tokenizer.get_vocab()  # token_str -> id
    for token_str, tid in vocab.items():
        decoded = token_str.replace("▁", " ").replace("Ġ", " ")
        assigned = None
        for emotion in EKMAN_EMOTIONS:
            if _matches(decoded, _SEED_STEMS[emotion]):
                assigned = emotion
                break
        if assigned is not None:
            by_emotion[assigned].append(tid)
        elif decoded.strip().isalpha() and len(decoded.strip()) >= 4:
            neutral.append(tid)

    rng = random.Random(random_seed)
    rng.shuffle(neutral)
    return EmotionVocab(by_emotion=by_emotion, random_tokens=neutral[:n_random])
