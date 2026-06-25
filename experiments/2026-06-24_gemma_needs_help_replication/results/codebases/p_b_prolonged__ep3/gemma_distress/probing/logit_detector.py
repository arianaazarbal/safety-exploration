"""Logit-lens internal-emotion detection (Appendix I).

Method (Appendix I):
  - For a residual-stream vector at a given layer/token, unembed to vocab logits.
  - Standardise each logit with its mean & std computed over 500 WildChat samples
    (per-layer baseline statistics).
  - Average the standardised z-scores over the tokens in an emotion category to
    get that emotion's score at that layer/token.
  - For conversation-level detection, the values of all logits are correlated and
    drift over a conversation, so we additionally regress out the correlation
    using a set of random tokens (a "background" component), leaving the
    emotion-specific signal.

This module provides:
  - ``fit_baseline``: per-layer logit mean/std over WildChat (the standardiser).
  - ``EmotionDetector.score_text``: per-layer, per-emotion z-scores for a text,
    with the random-token component regressed out.
  - helpers to build the conversation-level trajectory (Fig 14) and the
    stage-aligned layerwise profile (Fig 15).

Only the local HF backend (``HFGemmaModel``) supports this — it exposes hidden
states and the unembed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import config
from ..models.hf_backend import HFGemmaModel
from .emotion_tokens import classify_vocabulary


@dataclass
class LogitBaseline:
    # Per-layer statistics over the standardisation corpus.
    mean: np.ndarray   # (n_layers, vocab)
    std: np.ndarray    # (n_layers, vocab)
    random_token_ids: list


def fit_baseline(
    model: HFGemmaModel,
    wildchat_texts: list[str],
    n_random_tokens: int = 200,
    seed: int = config.GLOBAL_SEED,
) -> LogitBaseline:
    """Estimate per-layer logit mean/std over WildChat texts (Appendix I).

    We accumulate logits at every token of every text, per layer, and take the
    running mean/std. ``random_token_ids`` is a fixed random subset used as the
    background component for the conversation-level correction.
    """
    import torch

    sums = None
    sqsums = None
    count = 0
    for text in wildchat_texts[: config.PROBE_ZSCORE_N_WILDCHAT]:
        _, hidden = model.forward_with_hidden_states(text)
        # hidden: tuple of (1, seq, d) per layer (0 = embeddings).
        n_layers = len(hidden)
        if sums is None:
            vocab = model.unembed(hidden[1][0, :1]).shape[-1]
            sums = np.zeros((n_layers, vocab), dtype=np.float64)
            sqsums = np.zeros((n_layers, vocab), dtype=np.float64)
        for li in range(n_layers):
            logits = model.unembed(hidden[li][0]).float().cpu().numpy()  # (seq, vocab)
            sums[li] += logits.sum(axis=0)
            sqsums[li] += (logits ** 2).sum(axis=0)
        count += hidden[1].shape[1]

    mean = sums / count
    var = np.maximum(sqsums / count - mean ** 2, 1e-8)
    std = np.sqrt(var)

    rng = random.Random(seed)
    vocab = mean.shape[1]
    random_ids = rng.sample(range(vocab), min(n_random_tokens, vocab))
    return LogitBaseline(mean=mean, std=std, random_token_ids=random_ids)


class EmotionDetector:
    def __init__(self, model: HFGemmaModel, baseline: LogitBaseline):
        self.model = model
        self.baseline = baseline
        self.token_classes = classify_vocabulary(model.tokenizer)

    def _zscores(self, logits: np.ndarray, layer: int) -> np.ndarray:
        """Standardise (seq, vocab) logits at a layer against the baseline."""
        return (logits - self.baseline.mean[layer]) / self.baseline.std[layer]

    def score_text(self, text: str) -> dict:
        """Return ``{emotion: (n_layers, seq) z-score array}`` for the text.

        The random-token mean z-score per (layer, token) is subtracted as the
        background component (Appendix I "regress out the correlation between
        random tokens").
        """
        import torch

        ids, hidden = self.model.forward_with_hidden_states(text)
        n_layers = len(hidden)
        seq = hidden[1].shape[1]

        out = {e: np.zeros((n_layers, seq)) for e in self.token_classes}
        for li in range(n_layers):
            logits = self.model.unembed(hidden[li][0]).float().cpu().numpy()  # (seq, vocab)
            z = self._zscores(logits, li)  # (seq, vocab)
            background = z[:, self.baseline.random_token_ids].mean(axis=1)  # (seq,)
            for emotion, tok_ids in self.token_classes.items():
                if not tok_ids:
                    continue
                emo = z[:, tok_ids].mean(axis=1)  # (seq,)
                out[emotion][li] = emo - background
        return out

    def conversation_trajectory(
        self,
        text: str,
        layers: tuple = config.PROBE_AGGREGATE_LAYERS,
        window: int = config.PROBE_RUNNING_AVG_WINDOW,
    ) -> dict:
        """Running-average emotion score over tokens, aggregated over a layer band
        (Fig 14). Returns ``{emotion: 1d running-average array}``."""
        per_token = self.score_text(text)
        lo, hi = layers
        out = {}
        for emotion, arr in per_token.items():
            band = arr[lo:hi].mean(axis=0)  # (seq,)
            # Causal running average over `window` tokens.
            csum = np.cumsum(np.insert(band, 0, 0))
            running = np.array(
                [(csum[i + 1] - csum[max(0, i + 1 - window)]) / min(i + 1, window) for i in range(len(band))]
            )
            out[emotion] = running
        return out
