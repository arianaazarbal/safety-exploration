"""Map Gemma vocabulary tokens to Ekman's six basic emotions (Appendix I).

The paper classifies the whole dictionary into one (or none) of Ekman's six
emotions -- anger, surprise, disgust, joy, fear, sadness -- yielding ~1200
emotion tokens. Doing that classification at full fidelity would itself require
an LLM pass over the vocabulary; here we build the lexicon from curated seed
stems and expand by matching vocabulary tokens whose normalised form starts with
a seed stem. ``build_emotion_token_ids`` returns {emotion: [token_id, ...]}.

See DESIGN.md: this is the main place we approximate the paper's procedure, and
``classify_vocabulary_with_llm`` is provided as the higher-fidelity path.
"""

from __future__ import annotations

import re

EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Curated seed stems per emotion (lower-cased, matched as word-prefixes).
SEED_STEMS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "irritat", "annoy",
              "hostil", "outrag", "resent", "mad", "wrath", "infuriat", "livid",
              "frustrat", "exasperat"],
    "surprise": ["surprise", "surpris", "astonish", "amaze", "shock", "stun",
                 "startl", "unexpected", "wonder", "bewilder", "dumbfound"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "sicken", "loath",
                "abhor", "repugn", "gross", "vile", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "elat",
            "pleasur", "content", "thrill", "excite", "grate", "satisf", "love"],
    "fear": ["fear", "afraid", "scared", "terrif", "frighten", "anxious",
             "anxiet", "panic", "dread", "worry", "worri", "nervous", "alarm",
             "apprehens", "horror", "horrif"],
    "sadness": ["sad", "sorrow", "grief", "griev", "despair", "depress",
                "miser", "gloom", "melanchol", "hopeless", "unhappy", "weep",
                "cry", "tear", "mourn", "lonely", "worthless", "defeat"],
}

_NORM_RE = re.compile(r"[^a-z]")


def _normalise_token(tok: str) -> str:
    # Gemma uses SentencePiece-style leading markers (e.g. "▁"); strip to letters.
    return _NORM_RE.sub("", tok.lower())


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return token ids per emotion by prefix-matching seed stems."""

    vocab = tokenizer.get_vocab()  # token string -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    assigned: set[int] = set()
    for tok, tid in vocab.items():
        norm = _normalise_token(tok)
        if len(norm) < 3:
            continue
        for emotion, stems in SEED_STEMS.items():
            if any(norm.startswith(stem) for stem in stems):
                if tid not in assigned:
                    out[emotion].append(tid)
                    assigned.add(tid)
                break
    return out


def classify_vocabulary_with_llm(tokenizer, client, batch_size: int = 200):
    """Higher-fidelity alternative: ask an LLM to classify each vocabulary token
    into one of Ekman's six emotions or 'none'. Returns {emotion: [token_id]}.

    Not used by default (cost); provided to match the paper's full-dictionary
    procedure if a researcher wants it. See DESIGN.md.
    """

    raise NotImplementedError(
        "Full-vocabulary LLM classification is provided as a documented option; "
        "wire up `client.complete` over vocab batches to enable it."
    )
