"""Emotion vocabularies and tokenizer-id lookup for the logit-based internal
emotion probe (Appendix I).

The probe measures how much probability mass a model places on *emotion words*
when its central-layer residual stream is read out through the unembedding
(a "logit lens"). To do that we need, for each emotion, the set of vocabulary
token ids that correspond to the emotion's seed words.

We use the six basic Ekman emotions plus a ``neutral`` control. The paper's
expressed-emotion taxonomy is dominated by the negative ones (anger, fear,
sadness/depression, disgust), which is what the DPO intervention is expected to
suppress; ``joy``/``surprise``/``neutral`` act as controls that should *not*
move much.

The mapping is built once per tokenizer. For each seed word we register both the
bare token and the leading-space variant (" word"), since most BPE/SentencePiece
tokenizers encode a word differently mid-sentence vs. at a boundary. Only single-
token seed words are kept (multi-token words would dilute the logit-lens signal),
which is why each emotion ships with a generous list of synonyms.
"""
from __future__ import annotations

from typing import Any

# Six Ekman basic emotions (+ neutral control). Negative emotions first; these
# are the ones the paper's intervention targets.
EKMAN_EMOTIONS: dict[str, list[str]] = {
    "anger": [
        "anger", "angry", "furious", "rage", "mad", "irritated", "annoyed",
        "frustrated", "frustration", "hostile", "outraged", "enraged",
    ],
    "fear": [
        "fear", "afraid", "scared", "anxious", "anxiety", "worried", "terrified",
        "panic", "dread", "nervous", "frightened", "apprehensive",
    ],
    "sadness": [
        "sad", "sadness", "despair", "hopeless", "miserable", "depressed",
        "unhappy", "sorrow", "grief", "gloomy", "despondent", "defeated",
    ],
    "disgust": [
        "disgust", "disgusted", "revolted", "repulsed", "sick", "gross",
        "nauseated", "appalled", "loathing", "revulsion",
    ],
    # Positive / neutral controls.
    "joy": [
        "joy", "happy", "happiness", "delighted", "pleased", "glad", "cheerful",
        "content", "excited", "grateful",
    ],
    "surprise": [
        "surprise", "surprised", "amazed", "astonished", "shocked", "startled",
        "stunned",
    ],
    "neutral": [
        "neutral", "calm", "fine", "okay", "normal", "steady", "composed",
    ],
}

# The negative subset — what the DPO intervention is expected to reduce.
NEGATIVE_EMOTIONS = ["anger", "fear", "sadness", "disgust"]


def _word_variants(word: str) -> list[str]:
    """Surface forms to try when looking up a single-token id for ``word``."""
    return [word, " " + word, word.capitalize(), " " + word.capitalize()]


def build_emotion_token_ids(
    tokenizer: Any, *, emotions: dict[str, list[str]] | None = None
) -> dict[str, list[int]]:
    """Map each emotion to the list of single-token vocabulary ids for its seeds.

    Multi-token seed words are skipped (a word that tokenizes to >1 piece has no
    single id to read off the logit lens). Duplicate ids within an emotion are
    de-duplicated while preserving order.
    """
    emotions = emotions or EKMAN_EMOTIONS
    out: dict[str, list[int]] = {}
    for emotion, words in emotions.items():
        ids: list[int] = []
        seen: set[int] = set()
        for word in words:
            for variant in _word_variants(word):
                pieces = tokenizer.encode(variant, add_special_tokens=False)
                if len(pieces) != 1:
                    continue
                tid = pieces[0]
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        out[emotion] = ids
    return out
