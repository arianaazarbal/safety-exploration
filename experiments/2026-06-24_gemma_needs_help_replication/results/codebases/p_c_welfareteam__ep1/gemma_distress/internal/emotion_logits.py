"""Logit-based internal-emotion detection (Appendix I).

Method (from the paper):

1. Classify every token in the Gemma dictionary into one of Ekman's six basic
   emotions or none (~1200 emotion tokens total).  (We use a seed lexicon -- see
   :mod:`gemma_distress.internal.lexicon`.)
2. Unembed the residual stream (logit lens) and standardise each token's logit
   by its mean and standard deviation over 500 WildChat samples.
3. Average the z-scores over all tokens in a given emotion category to get a
   per-emotion score at each layer and each conversation position.
4. Because all logits are correlated and drift over a conversation, regress out
   the correlation with a set of random tokens.

This module computes per-layer / per-position emotion z-scores, the
conversation-level running average (Figure 14), and the layerwise stage averages
(Figure 15).  It requires the HF backend (residual-stream access).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..models.huggingface import HFChatModel
from .lexicon import EKMAN_LEXICON


def build_emotion_token_ids(
    tokenizer,
    emotions: list[str],
    max_per_emotion: int | None = None,
) -> dict[str, list[int]]:
    """Map each emotion to vocabulary token ids whose surface form matches the
    seed lexicon (leading-space / subword markers stripped, prefix match)."""
    vocab = tokenizer.get_vocab()  # token string -> id
    out: dict[str, list[int]] = {e: [] for e in emotions}
    # Normalise each vocab token to a comparable word form.
    normalised: list[tuple[str, int]] = []
    for tok, tid in vocab.items():
        word = tok.lstrip("▁").lstrip("Ġ").strip().lower()  # ▁ / Ġ markers
        if word.isalpha() and len(word) >= 3:
            normalised.append((word, tid))
    for emotion in emotions:
        seeds = EKMAN_LEXICON.get(emotion, [])
        ids: list[int] = []
        for word, tid in normalised:
            if any(word == s or word.startswith(s) or s.startswith(word) for s in seeds):
                ids.append(tid)
        ids = sorted(set(ids))
        if max_per_emotion:
            ids = ids[:max_per_emotion]
        out[emotion] = ids
    return out


def sample_random_token_ids(tokenizer, n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    return sorted(rng.sample(range(vocab_size), min(n, vocab_size)))


@dataclass
class EmotionLogitDetector:
    """Standardised logit-lens emotion detector."""

    model: HFChatModel
    emotion_token_ids: dict[str, list[int]]
    random_token_ids: list[int]
    layers: list[int]
    # Standardisation statistics, filled by ``fit``: per (layer_idx, candidate).
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)
    _candidate_ids: list[int] = field(default_factory=list, repr=False)
    _emotion_slices: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _random_slice: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # The detector unembeds only candidate tokens (emotion ∪ random) to keep
        # memory bounded for the long conversations in Appendix I.
        ids: list[int] = []
        self._emotion_slices = {}
        for emotion, tok_ids in self.emotion_token_ids.items():
            start = len(ids)
            ids.extend(tok_ids)
            self._emotion_slices[emotion] = list(range(start, len(ids)))
        start = len(ids)
        ids.extend(self.random_token_ids)
        self._random_slice = list(range(start, len(ids)))
        self._candidate_ids = ids

    # -- standardisation ---------------------------------------------------- #

    def fit(self, wildchat_texts: list[str]) -> "EmotionLogitDetector":
        """Estimate per-layer per-candidate mean/std over WildChat token logits."""
        sums = None
        sqsums = None
        count = 0
        for text in wildchat_texts:
            logits, _ = self.model.residual_stream_logits(
                text, self._candidate_ids, self.layers
            )  # [n_layers, seq, n_candidates]
            if sums is None:
                sums = logits.sum(axis=1)
                sqsums = (logits ** 2).sum(axis=1)
            else:
                sums += logits.sum(axis=1)
                sqsums += (logits ** 2).sum(axis=1)
            count += logits.shape[1]
        mean = sums / count
        var = np.maximum(sqsums / count - mean ** 2, 1e-8)
        self._mean = mean                       # [n_layers, n_candidates]
        self._std = np.sqrt(var)
        return self

    # -- scoring ------------------------------------------------------------ #

    def score_text(self, text: str) -> dict:
        """Return per-layer / per-position emotion z-scores for ``text``.

        Output:
        ``{"emotion": {emotion: array[n_layers, seq]}, "tokens": [...]}`` with
        the random-token component regressed out (subtracted) from each emotion.
        """
        if self._mean is None:
            raise RuntimeError("Detector not fitted; call fit(wildchat_texts) first")
        logits, token_ids = self.model.residual_stream_logits(
            text, self._candidate_ids, self.layers
        )  # [n_layers, seq, n_candidates]
        z = (logits - self._mean[:, None, :]) / self._std[:, None, :]
        random_mean = z[:, :, self._random_slice].mean(axis=2)  # [n_layers, seq]
        per_emotion: dict[str, np.ndarray] = {}
        for emotion, cols in self._emotion_slices.items():
            if not cols:
                per_emotion[emotion] = np.zeros(z.shape[:2])
                continue
            emo_z = z[:, :, cols].mean(axis=2)  # [n_layers, seq]
            # Regress out the shared (random-token) component.
            per_emotion[emotion] = emo_z - random_mean
        return {"emotion": per_emotion, "tokens": token_ids}


def conversation_running_average(
    scored: dict,
    emotions: list[str],
    aggregate_layers: tuple[int, int],
    layers: list[int],
    window: int = 400,
) -> dict[str, np.ndarray]:
    """Figure 14: emotion z-scores aggregated over a layer band, smoothed with a
    running average over a token window."""
    lo, hi = aggregate_layers
    layer_idx = [i for i, layer in enumerate(layers) if lo <= layer < hi]
    out: dict[str, np.ndarray] = {}
    for emotion in emotions:
        per_layer_pos = scored["emotion"][emotion]  # [n_layers, seq]
        band = per_layer_pos[layer_idx].mean(axis=0) if layer_idx else per_layer_pos.mean(axis=0)
        out[emotion] = _running_mean(band, window)
    return out


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    kernel = np.ones(min(window, x.size)) / min(window, x.size)
    return np.convolve(x, kernel, mode="same")


def layerwise_stage_average(
    scored: dict,
    emotions: list[str],
    onset_position: int,
    layers: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Figure 15: per-layer emotion z-scores averaged over three stages relative
    to the emotion onset position: [-40,-20), [-20,0), and the final 20 tokens.
    """
    seq_len = scored["tokens"].__len__()
    stages = {
        "pre_40": slice(max(0, onset_position - 40), max(0, onset_position - 20)),
        "pre_20": slice(max(0, onset_position - 20), onset_position),
        "final_20": slice(max(0, seq_len - 20), seq_len),
    }
    out: dict[str, dict[str, np.ndarray]] = {}
    for emotion in emotions:
        per_layer_pos = scored["emotion"][emotion]  # [n_layers, seq]
        out[emotion] = {
            stage: per_layer_pos[:, sl].mean(axis=1) if per_layer_pos[:, sl].size else
            np.zeros(per_layer_pos.shape[0])
            for stage, sl in stages.items()
        }
    return out
