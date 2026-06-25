"""Ekman emotion lexicon for internal-emotion detection (Appendix I).

The paper classifies every token in the Gemma vocabulary as describing one (or
none) of Ekman's six basic emotions — anger, surprise, disgust, joy, fear,
sadness — yielding ~1200 emotion tokens (~200 per emotion). The paper does not
state the exact classification procedure; we use a curated seed lexicon of
emotion stems per category and select vocabulary tokens whose normalised form
starts with one of the stems, capped at `tokens_per_emotion`. See DESIGN.md
("Internal emotion detection") for the rationale and the gap this fills.
"""
from __future__ import annotations

from typing import Dict, List

# Seed stems per Ekman category. Matching is done on the *stem* so morphological
# variants ("frustrat" -> frustrated/frustrating/frustration) are captured.
EKMAN_STEMS: Dict[str, List[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "resent", "outrage", "infuriat", "mad", "wrath", "indignant", "irate",
        "aggrav", "exasperat", "incens", "seeth", "livid", "spite", "hatred",
        "hate", "loath", "contempt", "bitter", "frustrat", "pissed", "grr", "argh",
    ],
    "surprise": [
        "surpris", "astonish", "amaz", "shock", "startl", "stunn", "astound",
        "bewilder", "dumbfound", "flabbergast", "unexpected", "wow", "whoa",
        "incredul", "speechless", "taken aback", "jolt", "gasp", "wonder",
        "baffl", "perplex", "confus", "stupef",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "repugn", "nause", "sicken", "loathe",
        "abhor", "detest", "gross", "vile", "repell", "distast", "aversion",
        "queasy", "yuck", "ew", "appall", "horrid", "putrid", "foul", "nasty",
        "offens", "odious",
    ],
    "joy": [
        "joy", "happy", "happi", "delight", "glad", "cheer", "pleas", "content",
        "elat", "ecstat", "thrill", "jubil", "gleeful", "merry", "blissful",
        "satisf", "grateful", "excit", "enthus", "optimis", "hopeful", "love",
        "wonderful", "great", "fantastic", "smile", "laugh", "celebrat", "yay",
    ],
    "fear": [
        "fear", "afraid", "scared", "scare", "terrif", "frighten", "panic",
        "anxious", "anxiet", "dread", "horror", "horrif", "alarm", "nervous",
        "worri", "apprehens", "phobi", "petrif", "trepidat", "uneasy", "tense",
        "spook", "intimidat", "threat", "danger",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "miser", "gloom", "melanchol",
        "depress", "mourn", "weep", "cry", "tear", "heartbreak", "hopeless",
        "despond", "dismal", "lonely", "loneli", "forlorn", "wretch", "dejected",
        "downcast", "unhappy", "regret", "disappoint", "anguish", "suffer",
        "worthless", "defeat", "giving up", "give up",
    ],
}


def normalise_token(tok: str) -> str:
    """Normalise a raw tokenizer piece: strip leading sentencepiece marker
    (chr 0x2581 '▁') and surrounding whitespace, lowercase."""
    return tok.replace("▁", " ").strip().lower()


def classify_token(norm_tok: str) -> List[str]:
    """Return the list of Ekman categories a normalised token belongs to.

    A token is assigned to a category if it *starts with* one of the category's
    stems (length-2+ token to avoid spurious single letters).
    """
    if len(norm_tok) < 3:
        return []
    cats = []
    for emotion, stems in EKMAN_STEMS.items():
        if any(norm_tok.startswith(stem) for stem in stems):
            cats.append(emotion)
    return cats


def build_emotion_token_ids(
    tokenizer,
    emotions: List[str],
    tokens_per_emotion: int = 200,
) -> Dict[str, List[int]]:
    """Scan the tokenizer vocab and return token ids per emotion (capped).

    Tokens assigned to more than one category are dropped (we want tokens that
    describe exactly one emotion, per the paper's "one or none" framing).
    """
    vocab = tokenizer.get_vocab()  # token string -> id
    by_emotion: Dict[str, List[int]] = {e: [] for e in emotions}
    for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        norm = normalise_token(tok)
        cats = [c for c in classify_token(norm) if c in emotions]
        if len(cats) != 1:
            continue
        e = cats[0]
        if len(by_emotion[e]) < tokens_per_emotion:
            by_emotion[e].append(tid)
    return by_emotion
