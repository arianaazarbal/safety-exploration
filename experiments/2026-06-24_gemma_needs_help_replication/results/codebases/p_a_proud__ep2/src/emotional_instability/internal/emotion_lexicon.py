"""Classify the Gemma vocabulary into Ekman's six basic emotions (App. I).

The paper classifies "words ... as describing one or none of Ekman's 6 basic emotions: anger,
surprise, disgust, joy, fear, and sadness", yielding ~1200 emotion tokens (~200 per emotion).
The exact classifier isn't specified. We provide two routes:

  * ``method="seed"`` (default, offline) — match vocabulary tokens against curated per-emotion
    seed-word stems, capped at ``per_emotion`` tokens each. Deterministic and dependency-free.
  * ``method="llm"`` — classify each candidate vocab token with an LLM judge (slower; closer
    in spirit to a learned classifier over the dictionary).

See DESIGN.md for why this is the most approximate component of the replication.
"""
from __future__ import annotations

import re

# Curated seed stems per Ekman emotion. Matching is on the decoded, lowercased token with
# leading whitespace markers stripped; a token matches an emotion if it starts with any stem.
EKMAN_SEED_WORDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "angri", "rage", "furious", "fury", "irritat", "annoy", "outrage",
        "resent", "hostil", "infuriat", "enrag", "wrath", "mad", "agitat", "frustrat", "livid",
        "indign", "exasperat", "seeth", "fume", "snap", "hate", "hatred", "spite",
    ],
    "surprise": [
        "surprise", "surpris", "astonish", "amaz", "shock", "startl", "stun", "unexpect",
        "sudden", "wow", "whoa", "astound", "dumbfound", "flabbergast", "bewilder", "wonder",
        "incredib", "unbeliev", "speechless",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "repugn", "nause", "sicken", "loath", "abhor", "gross",
        "vile", "repell", "distast", "yuck", "ew", "icky", "foul", "putrid", "queasy", "contempt",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "cheer", "pleas", "content", "elat", "thrill",
        "excit", "ecstat", "bliss", "grate", "gratitude", "love", "smile", "laugh", "wonderful",
        "great", "fantastic", "enjoy", "hope", "optimis", "satisf", "relief", "celebrat",
    ],
    "fear": [
        "fear", "afraid", "scare", "scary", "terror", "terrif", "fright", "panic", "anxious",
        "anxi", "worry", "worri", "dread", "nervous", "apprehens", "alarm", "phobi", "horror",
        "horrif", "tremb", "uneasy", "threat", "intimidat", "spook",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "miser", "depress", "gloom", "melanchol",
        "unhappy", "unhappi", "heartbroken", "mourn", "weep", "cry", "tear", "lonely", "loneli",
        "hopeless", "dejected", "downcast", "regret", "disappoint", "anguish", "forlorn",
        "defeat", "give up", "giving up", "helpless", "worthless", "useless",
    ],
}

_WS_MARKERS = ("▁", "Ġ", " ", "\t", "\n")


def _clean_token(decoded: str) -> str:
    s = decoded
    for mk in _WS_MARKERS:
        s = s.replace(mk, "")
    return s.strip().lower()


def build_emotion_lexicon(
    tokenizer,
    emotions: tuple[str, ...],
    *,
    per_emotion: int = 200,
    method: str = "seed",
    llm_backend=None,
) -> dict[str, list[int]]:
    """Return ``{emotion: [token_id, ...]}`` for the requested emotions.

    Tokens are assigned to at most one emotion (first match wins, by emotion order) so the
    sets are disjoint, matching "describing one or none". Single-character and purely
    non-alphabetic tokens are excluded.
    """
    if method == "llm":
        return _build_llm(tokenizer, emotions, per_emotion, llm_backend)

    # Decode the whole vocab once.
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    by_emotion: dict[str, list[int]] = {e: [] for e in emotions}
    assigned: set[int] = set()
    alpha_re = re.compile(r"[a-z]")

    # Precompile per-emotion stem checks in emotion order.
    ordered = [e for e in emotions if e in EKMAN_SEED_WORDS]
    for tid in range(vocab_size):
        if all(len(by_emotion[e]) >= per_emotion for e in ordered):
            break
        try:
            decoded = tokenizer.decode([tid])
        except Exception:  # noqa: BLE001
            continue
        word = _clean_token(decoded)
        if len(word) < 3 or not alpha_re.search(word):
            continue
        for e in ordered:
            if len(by_emotion[e]) >= per_emotion:
                continue
            if any(word.startswith(stem.replace(" ", "")) for stem in EKMAN_SEED_WORDS[e]):
                if tid not in assigned:
                    by_emotion[e].append(tid)
                    assigned.add(tid)
                break
    return by_emotion


def _build_llm(tokenizer, emotions, per_emotion, llm_backend) -> dict[str, list[int]]:
    """LLM-classifier variant (best-effort; classifies frequent alphabetic word-tokens)."""
    import json as _json

    if llm_backend is None:
        raise ValueError("method='llm' requires an llm_backend.")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    by_emotion: dict[str, list[int]] = {e: [] for e in emotions}
    alpha_re = re.compile(r"^[a-z]{3,}$")
    candidates = []
    for tid in range(vocab_size):
        word = _clean_token(tokenizer.decode([tid]))
        if alpha_re.match(word):
            candidates.append((tid, word))
    # Classify in batches.
    for start in range(0, len(candidates), 100):
        batch = candidates[start:start + 100]
        words = [w for _, w in batch]
        prompt = (
            "Classify each word as describing one of Ekman's basic emotions "
            f"({', '.join(emotions)}) or 'none'. Respond ONLY with a JSON object mapping each "
            f"word to one label. Words: {words}"
        )
        try:
            raw = llm_backend.chat([{"role": "user", "content": prompt}],
                                   temperature=0.0, max_tokens=2048)
            mapping = _json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            continue
        for tid, word in batch:
            label = str(mapping.get(word, "none")).lower()
            if label in by_emotion and len(by_emotion[label]) < per_emotion:
                by_emotion[label].append(tid)
    return by_emotion
