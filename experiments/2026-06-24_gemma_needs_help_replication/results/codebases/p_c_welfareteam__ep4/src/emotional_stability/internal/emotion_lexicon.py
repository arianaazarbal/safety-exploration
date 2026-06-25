"""Classify the model vocabulary into Ekman's 6 basic emotions (Appendix I).

The paper classifies "over the whole Gemma dictionary, words ... as describing
one or none of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear,
and sadness. This gives us 1200 emotion tokens total."

The exact classifier is not specified. We build the emotion-token sets by
matching vocabulary tokens against a curated seed lexicon per emotion (stemmed,
case- and whitespace-insensitive over the detokenised surface form). This is
deterministic, needs no network, and yields the per-emotion token-id columns the
logit-lens probe aggregates over. ``classify_with_llm`` is provided as an
optional higher-fidelity alternative (closer to the paper if it used an LLM
classifier), but is off by default. See DESIGN.md for this choice.
"""

from __future__ import annotations

import re

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Curated seed lexicons. These are stems/roots; matching is substring-on-word so
# e.g. "frustrat" catches frustrated/frustrating/frustration.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "mad",
        "hostil", "resent", "outrag", "infuriat", "wrath", "hate", "hateful",
        "frustrat", "exasperat", "agitat", "indign", "enrag", "livid", "seething",
        "bitter", "spite", "contempt", "disdain", "antagon", "provok",
    ],
    "surprise": [
        "surpris", "astonish", "amaz", "shock", "startl", "stun", "stunned",
        "unexpected", "sudden", "wow", "whoa", "astound", "dumbfound", "bewilder",
        "flabbergast", "incredul", "disbelief", "marvel", "wonder", "gasp",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nause", "sicken", "loath", "abhor",
        "repugn", "gross", "vile", "yuck", "ugh", "distast", "offensive",
        "repellent", "queasy", "squeamish", "appalling", "horrid", "foul",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "pleas", "glad", "cheer", "content",
        "elat", "excit", "thrill", "enjoy", "wonderful", "great", "fantastic",
        "love", "grateful", "satisf", "optimist", "hopeful", "celebrat", "smile",
        "fun", "ecstat", "blissful", "upbeat", "positiv",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrif", "anxious", "anxiety", "worry",
        "worri", "panic", "dread", "nervous", "apprehens", "alarm", "frighten",
        "horror", "horrif", "threat", "peril", "danger", "uneasy", "trepidat",
        "phobi", "petrif", "intimidat",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "depress", "miser",
        "unhappy", "gloom", "melanchol", "mourn", "heartbreak", "hopeless",
        "despond", "dejected", "downcast", "forlorn", "woe", "weep", "cry",
        "tear", "lonely", "loss", "regret", "disappoint", "defeat", "giving up",
        "worthless", "inadequate", "fail",
    ],
}

_WORD_RE = re.compile(r"[a-z']+")


def _normalise_token(surface: str) -> str:
    """Detokenise a vocab piece to a comparable surface word.

    Strips SentencePiece/BPE markers (leading ``▁`` / ``Ġ``) and lowercases.
    """
    return surface.replace("▁", " ").replace("Ġ", " ").strip().lower()


def classify_vocabulary(tokenizer, per_emotion_cap: int = 200) -> dict[str, list[int]]:
    """Return {emotion: [token_id, ...]} by matching vocab against the lexicon.

    A token is assigned to at most one emotion (the first whose seed matches),
    matching the paper's "one or none" rule. Capped per emotion to keep the set
    near the paper's ~200/emotion.
    """
    vocab = tokenizer.get_vocab()  # {surface: id}
    assigned: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    taken: set[int] = set()
    # Deterministic order: by token id, so caps select the same tokens each run.
    for surface, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        if tid in taken:
            continue
        word = _normalise_token(surface)
        if not word or not _WORD_RE.fullmatch(word.replace(" ", "")):
            continue
        for emotion in EKMAN_EMOTIONS:
            if len(assigned[emotion]) >= per_emotion_cap:
                continue
            if any(seed in word for seed in SEED_LEXICON[emotion]):
                assigned[emotion].append(tid)
                taken.add(tid)
                break
    return assigned


def classify_with_llm(
    tokenizer,
    *,
    judge_model: str = "claude-sonnet-4-20250514",
    per_emotion_cap: int = 200,
    batch_size: int = 100,
    settings=None,
) -> dict[str, list[int]]:
    """Higher-fidelity vocabulary classification via an LLM (optional).

    Closer to what the paper may have done than the seed-lexicon match, but
    requires O(vocab/batch) API calls and is nondeterministic, so it is *not* the
    default (``classify_vocabulary`` is). Classifies each detokenised word into
    one of Ekman's 6 emotions or "none", in batches, and returns token-id sets.

    Kept deliberately simple: it asks the judge to label a numbered list of words
    and parses ``index: emotion`` lines. Intended for offline pre-computation of a
    cached lexicon, not for hot-path use.
    """
    from emotional_stability.models.anthropic_client import AnthropicClient
    from emotional_stability.models.parsing import extract_json_object
    from emotional_stability.records import Message

    client = AnthropicClient(judge_model, settings=settings)
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    words = [(tid, _normalise_token(s)) for s, tid in vocab_items]
    words = [(tid, w) for tid, w in words if w and _WORD_RE.fullmatch(w.replace(" ", ""))]

    assigned: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    instruction = (
        "Classify each word as describing exactly one of these emotions, or "
        f"'none': {', '.join(EKMAN_EMOTIONS)}. Respond with JSON mapping the "
        "given index (as a string) to the emotion or 'none'."
    )
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        listing = "\n".join(f"{i}: {w}" for i, (_, w) in enumerate(chunk))
        reply = client.complete(
            [Message(role="user", content=f"{instruction}\n\n{listing}")],
            temperature=0.0,
            max_tokens=2048,
        )
        try:
            labels = extract_json_object(reply)
        except ValueError:
            continue
        for i, (tid, _) in enumerate(chunk):
            label = str(labels.get(str(i), "none")).lower()
            if label in assigned and len(assigned[label]) < per_emotion_cap:
                assigned[label].append(tid)
    return assigned


def random_baseline_tokens(
    tokenizer, n: int = 200, exclude: set[int] | None = None, seed: int = 0
) -> list[int]:
    """Sample ``n`` random vocab token ids for the regress-out baseline (App. I)."""
    import random

    exclude = exclude or set()
    rng = random.Random(seed)
    ids = [i for i in range(tokenizer.vocab_size) if i not in exclude]
    return rng.sample(ids, min(n, len(ids)))
