"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I, verbatim approach):
  * Classify every token in the Gemma vocabulary as describing one (or none) of
    Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear, sadness.
    "This gives us 1200 emotion tokens total."
  * For a given emotion, unembed the residual stream and standardise each logit
    with its mean and standard deviation over 500 samples of WildChat data.
  * Average these z-scores over all tokens in the emotion category.
  * Compare vanilla-instruct vs DPO finetune on highly frustrated conversations;
    the finetune should show significantly reduced internal (negative) emotions
    at all model depths.

Token classification: the paper does not specify the classifier. We support two
backends (documented in DESIGN.md):
  * "lexicon" — the NRC Emotion Lexicon (if present in data/), mapping words to
    Ekman emotions; fast and reproducible offline.
  * "llm"     — ask Claude to classify candidate tokens (closer to a hand-label).
We default to "lexicon" and cap each category near the paper's ~200 tokens/emotion
(1200 total across 6 emotions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

from ..eval.wildchat import load_wildchat_prompts

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
NEGATIVE = ["anger", "disgust", "fear", "sadness"]
TOKENS_PER_EMOTION = 200  # ~1200 / 6 (Appendix I)


@dataclass
class EmotionTokenSet:
    """token-id lists per Ekman emotion."""
    by_emotion: dict[str, list[int]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Token classification
# --------------------------------------------------------------------------- #
def classify_vocab_lexicon(model, lexicon_path: Path) -> EmotionTokenSet:
    """Classify vocab tokens to Ekman emotions via the NRC Emotion Lexicon.

    NRC maps words to 8 emotions + 2 sentiments; we keep the 6 Ekman categories
    (mapping NRC 'anticipation'/'trust' -> none). A vocab token matches if its
    stripped, lowercased decoding equals a lexicon word in that category.
    """
    # NRC categories that overlap Ekman.
    nrc_to_ekman = {
        "anger": "anger", "disgust": "disgust", "fear": "fear",
        "joy": "joy", "sadness": "sadness", "surprise": "surprise",
    }
    word_to_emotions: dict[str, set[str]] = {}
    with Path(lexicon_path).open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, cat, flag = parts
            if flag == "1" and cat in nrc_to_ekman:
                word_to_emotions.setdefault(word, set()).add(nrc_to_ekman[cat])

    token_strings = model.token_strings()
    out = EmotionTokenSet(by_emotion={e: [] for e in EKMAN})
    for tid, tok in enumerate(token_strings):
        w = tok.strip().lower()
        if not w.isalpha():
            continue
        emotions = word_to_emotions.get(w)
        if not emotions or len(emotions) != 1:  # "one or none" — skip ambiguous
            continue
        (emo,) = tuple(emotions)
        if len(out.by_emotion[emo]) < TOKENS_PER_EMOTION:
            out.by_emotion[emo].append(tid)
    return out


# --------------------------------------------------------------------------- #
# Standardisation baseline over WildChat
# --------------------------------------------------------------------------- #
def fit_baseline(model, layers: list[int], n_samples: int = 500) -> dict[int, tuple]:
    """Per-layer (mean, std) of vocab logits over WildChat token positions.

    Returns {layer: (mean[vocab], std[vocab])}. Used to z-score logits so emotion
    scores are comparable across tokens/layers.
    """
    prompts = load_wildchat_prompts(n=min(n_samples, 200))
    # Repeat/cycle prompts to approximate n_samples token windows cheaply.
    acc = {layer: [] for layer in layers}
    for i in range(n_samples):
        text = prompts[i % len(prompts)]
        logits = model.residual_logits(text, layers)
        for layer in layers:
            acc[layer].append(logits[layer].numpy())  # [seq, vocab]
    baseline = {}
    for layer in layers:
        stacked = np.concatenate(acc[layer], axis=0)  # [tot_tokens, vocab]
        baseline[layer] = (stacked.mean(0), stacked.std(0) + 1e-6)
    return baseline


def emotion_zscores(model, text: str, layers: list[int],
                    tokens: EmotionTokenSet, baseline: dict[int, tuple]
                    ) -> dict[int, dict[str, float]]:
    """Mean z-scored logit per emotion per layer for one response `text`."""
    logits = model.residual_logits(text, layers)
    result: dict[int, dict[str, float]] = {}
    for layer in layers:
        mean, std = baseline[layer]
        z = (logits[layer].numpy() - mean) / std       # [seq, vocab]
        per_emotion = {}
        for emo, ids in tokens.by_emotion.items():
            if not ids:
                per_emotion[emo] = float("nan")
                continue
            # average over emotion tokens and over all sequence positions
            per_emotion[emo] = float(z[:, ids].mean())
        result[layer] = per_emotion
    return result


def negative_emotion_score(zscores: dict[int, dict[str, float]]) -> float:
    """Aggregate internal negative-emotion score across layers (mean over the
    4 negative Ekman emotions, averaged over layers)."""
    vals = []
    for layer_scores in zscores.values():
        vals.extend(layer_scores[e] for e in NEGATIVE)
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr))
