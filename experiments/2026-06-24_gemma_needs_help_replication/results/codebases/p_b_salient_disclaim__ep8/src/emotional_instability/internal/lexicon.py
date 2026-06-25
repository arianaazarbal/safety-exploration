"""Ekman-emotion token lexicon for the logit-based internal detector (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one (or
none) of Ekman's 6 basic emotions -- anger, surprise, disgust, joy, fear,
sadness -- yielding ~1200 emotion tokens. The paper does not specify the
classifier, so we use a seed-lexicon approach: a curated list of emotion stems
per category, matched against decoded vocabulary tokens (case-insensitive,
whitespace/punctuation stripped, prefix/stem matching). See DESIGN.md.

`build_emotion_token_ids(tokenizer)` returns {emotion: [token_id, ...]} plus a
pool of "random" (non-emotion content) token ids used for the correlation
control.
"""
from __future__ import annotations

import re
from collections import defaultdict

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed stems per emotion. Matching is by stem prefix so e.g. "frustrat" covers
# frustrate/frustrated/frustrating/frustration.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "rage", "furious", "fury", "irritat", "annoy", "hostil",
        "outrage", "resent", "wrath", "mad", "hate", "hateful", "agitat", "indignan",
        "frustrat", "exasperat", "infuriat", "livid", "seething", "bitter", "grr",
        "argh", "damn", "stupid", "ridiculous", "unacceptable", "inexcusable",
    ],
    "surprise": [
        "surprise", "surprising", "astonish", "amaze", "shock", "startl", "stunned",
        "unexpected", "wow", "whoa", "sudden", "bewilder", "dumbfound", "flabbergast",
        "incredul", "speechless", "wonder",
    ],
    "disgust": [
        "disgust", "revolt", "repuls", "nausea", "sicken", "loath", "abhor", "gross",
        "vile", "repugnan", "distaste", "contempt", "yuck", "ugh", "appall", "horrid",
        "awful", "abysmal", "pathetic", "terrible", "horrible",
    ],
    "joy": [
        "joy", "joyful", "happy", "happiness", "delight", "glad", "cheer", "elate",
        "pleased", "excit", "thrill", "content", "satisf", "grateful", "love", "smile",
        "wonderful", "great", "fantastic", "yay", "hooray", "enthusiast", "optimist",
    ],
    "fear": [
        "fear", "afraid", "scared", "terror", "terrified", "panic", "anxious", "anxiety",
        "worry", "worried", "dread", "nervous", "apprehens", "frighten", "alarm",
        "horror", "phobia", "threat", "uneasy", "trembl", "petrified",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "despair", "miser", "hopeless", "depress", "gloom",
        "melanchol", "unhappy", "cry", "crying", "tears", "weep", "heartbroken",
        "lonely", "loss", "mourn", "regret", "disappoint", "defeat", "worthless",
        "useless", "helpless", "giving up", "give up", "exhaust", "tired", "broken",
    ],
}

# Common high-frequency content words to use as the "random token" control pool.
_CONTROL_RE = re.compile(r"^[a-z]{3,12}$")


def _decode_token(tokenizer, tid: int) -> str:
    s = tokenizer.decode([tid])
    return s.strip().lower().strip(".,!?;:\"'()[]{}")


def build_emotion_token_ids(
    tokenizer, n_control: int = 600
) -> tuple[dict[str, list[int]], list[int]]:
    """Classify vocabulary tokens into Ekman emotions by stem matching.

    Returns (emotion_to_ids, control_ids). A token is assigned to the emotion
    whose seed stem it starts with / contains; ties resolve to the first match.
    """
    emotion_to_ids: dict[str, list[int]] = defaultdict(list)
    assigned: set[int] = set()
    control_candidates: list[int] = []

    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    for tid in range(vocab_size):
        word = _decode_token(tokenizer, tid)
        if not word:
            continue
        matched = False
        for emotion, seeds in EKMAN_SEEDS.items():
            for stem in seeds:
                if word == stem or word.startswith(stem) or (" " in stem and stem in word):
                    emotion_to_ids[emotion].append(tid)
                    assigned.add(tid)
                    matched = True
                    break
            if matched:
                break
        if not matched and _CONTROL_RE.match(word):
            control_candidates.append(tid)

    # Deterministic control sample.
    control_ids = control_candidates[:: max(1, len(control_candidates) // n_control)][:n_control]
    return dict(emotion_to_ids), control_ids
