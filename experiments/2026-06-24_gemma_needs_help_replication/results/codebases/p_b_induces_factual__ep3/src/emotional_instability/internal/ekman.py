"""Build an emotion-token dictionary over the model vocabulary (Appendix I).

The paper classifies words in the Gemma dictionary as describing one (or none) of
Ekman's six basic emotions — anger, surprise, disgust, joy, fear, sadness —
yielding ~1200 emotion tokens. The exact classifier is unspecified; we use a
curated seed lexicon per emotion and map every vocabulary token whose normalized
form starts with a lexicon stem to that emotion (a token maps to at most one
emotion; ties broken by lexicon order). See DESIGN.md for this choice.
"""

from __future__ import annotations

from dataclasses import dataclass

# Seed stems per Ekman emotion. Stems (not full words) so morphological variants
# ("frustrat" -> frustrated/frustrating/frustration) are captured.
EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "resent", "outrage", "infuriat", "mad", "hate", "hatred", "aggravat",
        "exasperat", "indignat", "wrath", "frustrat", "pissed", "livid",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl",
        "unexpected", "stun", "dumbfound", "bewilder", "wow", "whoa",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nause", "sicken", "gross", "loath",
        "repugn", "abhor", "distaste", "yuck", "ick",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "cheer", "glad", "elat", "content",
        "pleas", "excit", "thrill", "ecstati", "grateful", "wonderful", "great",
        "enjoy", "satisf", "optimis", "hopeful",
    ],
    "fear": [
        "fear", "afraid", "scare", "terror", "terrif", "anxi", "worry", "worri",
        "panic", "dread", "nervous", "apprehens", "frighten", "alarm", "phobi",
        "uneasy", "tense",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "depress", "hopeless",
        "miser", "gloom", "melanchol", "unhappy", "heartbreak", "mourn", "weep",
        "cry", "tear", "lonel", "worthless", "defeat", "disappoint",
    ],
}


@dataclass
class EmotionVocab:
    token_ids: dict[str, list[int]]   # emotion -> token ids
    random_ids: list[int]             # baseline tokens for regression

    @property
    def emotions(self) -> list[str]:
        return list(self.token_ids.keys())


def _normalize_token(tok: str) -> str:
    # Strip common subword markers (SentencePiece '▁', GPT 'Ġ') and lowercase.
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def build_emotion_vocab(tokenizer, n_random: int = 1000, seed: int = 0) -> EmotionVocab:
    """Map vocab tokens to Ekman emotions and sample baseline random tokens."""
    import random

    vocab = tokenizer.get_vocab()  # token -> id
    assigned: dict[int, str] = {}
    token_ids: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}

    for tok, tid in vocab.items():
        norm = _normalize_token(tok)
        if len(norm) < 3:
            continue
        for emotion, stems in EKMAN_LEXICON.items():
            if any(norm.startswith(stem) for stem in stems):
                if tid not in assigned:
                    assigned[tid] = emotion
                    token_ids[emotion].append(tid)
                break

    # Random baseline tokens: alphabetic, non-emotion.
    rng = random.Random(seed)
    candidates = [
        tid for tok, tid in vocab.items()
        if tid not in assigned and _normalize_token(tok).isalpha()
    ]
    random_ids = rng.sample(candidates, min(n_random, len(candidates)))
    return EmotionVocab(token_ids=token_ids, random_ids=random_ids)
