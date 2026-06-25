"""Classify the Gemma vocabulary into Ekman's 6 basic emotions (Appendix I).

"Over the whole Gemma dictionary, words are classified as describing one or none
of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear, and sadness.
This gives us 1200 emotion tokens total."

Two classification paths:
  1. lexicon  (default, offline): match each vocab token against curated seed
     word lists (+ simple morphological stemming). Self-contained, deterministic.
  2. llm      (optional): ask Claude to label tokens in batches -- closer to the
     paper's "classified over the whole dictionary", but costs API calls.

Either way the result is cached to artifacts so probing runs are reproducible.
We also sample a set of random (emotion-neutral) tokens used to estimate and
regress out the common-mode logit drift in logit_lens.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DATA_DIR

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
CACHE_PATH = DATA_DIR / "emotion_tokens.json"

# Curated seed lexicons (lemmas). Vocab tokens are matched if they start with /
# contain a lemma after stripping the leading word-boundary marker.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
              "resent", "outrage", "mad", "wrath", "infuriat", "agitat", "frustrat",
              "exasperat", "indignan", "livid", "seething", "hate", "hatred"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "amazing", "shock",
                 "startl", "stun", "unexpected", "wow", "whoa", "sudden", "bewilder",
                 "dumbfound", "flabbergast"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken", "loath",
                "abhor", "repugnan", "vile", "yuck", "ew", "distaste", "contempt"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "glad", "cheer", "elat",
            "ecsta", "content", "pleased", "thrill", "excite", "wonderful", "great",
            "love", "grateful", "optimis", "smile", "celebrat"],
    "fear": ["fear", "afraid", "scared", "terror", "terrif", "panic", "anxi", "worri",
             "worry", "dread", "fright", "nervous", "alarm", "apprehens", "horror",
             "horrif", "phobia", "uneasy", "tremb", "petrif"],
    "sadness": ["sad", "sadness", "sorrow", "grief", "despair", "depress", "miser",
                "gloom", "melanchol", "hopeless", "heartbroken", "mourn", "cry",
                "tearful", "unhappy", "dejected", "despondent", "lonel", "regret",
                "disappoint", "hurt", "suffering", "anguish", "woe"],
}


def _strip_marker(tok: str) -> str:
    # Gemma/SentencePiece use the leading-space marker; normalise it out.
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def classify_lexicon(vocab: dict[str, int]) -> dict[str, list[int]]:
    """Return emotion -> list of token ids using the seed lexicon."""
    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    for tok, tid in vocab.items():
        word = _strip_marker(tok)
        if len(word) < 3 or not word.isalpha():
            continue
        for emotion, lemmas in SEED_LEXICON.items():
            if any(word.startswith(lem) or lem in word for lem in lemmas):
                out[emotion].append(tid)
                break
    return out


def classify_llm(vocab: dict[str, int], batch_size: int = 200) -> dict[str, list[int]]:
    """Ask Claude to label vocabulary tokens (optional, costs API calls)."""
    from ..backends import get_backend
    from ..config import JUDGE_GEN, ONSET_LABELLER
    from ..utils import extract_json

    backend = get_backend(ONSET_LABELLER)
    items = [(tok, tid) for tok, tid in vocab.items()
             if _strip_marker(tok).isalpha() and len(_strip_marker(tok)) >= 3]
    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        words = [_strip_marker(t) for t, _ in chunk]
        prompt = (
            "Classify each word as describing ONE of Ekman's six basic emotions "
            f"({', '.join(EKMAN)}) or 'none'. A word qualifies only if it directly "
            "denotes or strongly connotes that emotion.\n"
            f"Words: {json.dumps(words)}\n"
            'Respond with JSON {"labels": [<one of '
            f'{EKMAN + ["none"]}> per word, in order]}}.'
        )
        try:
            data = extract_json(backend.generate([{"role": "user", "content": prompt}], JUDGE_GEN).text)
            labels = data.get("labels", [])
            for (tok, tid), lab in zip(chunk, labels):
                if lab in out:
                    out[lab].append(tid)
        except Exception:  # noqa: BLE001
            continue
    return out


def build_emotion_tokens(tokenizer, method: str = "lexicon", n_random: int = 500,
                         seed: int = 0, use_cache: bool = True) -> dict:
    if use_cache and CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())

    vocab = tokenizer.get_vocab()  # token -> id
    classified = classify_llm(vocab) if method == "llm" else classify_lexicon(vocab)

    emotion_ids = {ids for e in classified for ids in classified[e]}
    rng = random.Random(seed)
    neutral_pool = [tid for tok, tid in vocab.items()
                    if _strip_marker(tok).isalpha() and tid not in emotion_ids]
    random_tokens = rng.sample(neutral_pool, min(n_random, len(neutral_pool)))

    result = {
        "method": method,
        "ekman": {e: sorted(classified[e]) for e in EKMAN},
        "counts": {e: len(classified[e]) for e in EKMAN},
        "total": sum(len(classified[e]) for e in EKMAN),
        "random_tokens": random_tokens,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    return result
