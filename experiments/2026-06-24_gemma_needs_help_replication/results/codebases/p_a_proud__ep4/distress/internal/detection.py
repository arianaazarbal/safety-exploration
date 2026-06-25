"""Logit-based internal-emotion detection (Paper Appendix I).

Method:
  * Build a baseline by unembedding the residual stream over ~500 WildChat
    samples and recording, for every (layer, emotion-token), the mean and std of
    its logit.
  * For a target conversation, z-score each emotion-token logit against that
    baseline, regress out the shared drift estimated from random (neutral) tokens,
    then average the z-scores over the tokens in each Ekman category to get an
    emotion score per layer (and per position, for trajectories).

Comparing the vanilla vs DPO model on the same frustrated responses shows whether
the intervention suppresses *internal* emotion (Appendix I, Figures 14/15).

Requires a local ``HFBackend`` (residual-stream access). NumPy-based after pulling
logits to CPU; baseline statistics use a streaming (Welford) accumulator to bound
memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.hf_backend import HFBackend
from .emotion_tokens import EKMAN_EMOTIONS, EmotionVocab, build_emotion_vocab


@dataclass
class EmotionScores:
    """Per-layer emotion z-scores for one text (averaged over token positions)."""

    by_emotion: dict[str, np.ndarray]   # emotion -> (n_layers,)
    # Per-position, per-layer trajectory for the dominant queries (optional use).
    trajectory: dict[str, np.ndarray] | None = None


class InternalEmotionDetector:
    def __init__(self, model: HFBackend, emotion_vocab: EmotionVocab | None = None):
        if not isinstance(model, HFBackend):
            raise TypeError("Internal detection requires a local HFBackend model.")
        self.model = model
        self.vocab = emotion_vocab or build_emotion_vocab(model.tokenizer)
        # Combined query token list: emotion tokens then random tokens.
        self.emotion_tokens = self.vocab.all_emotion_tokens
        self.random_tokens = self.vocab.random_tokens
        self.query_tokens = self.emotion_tokens + self.random_tokens
        self._n_emotion = len(self.emotion_tokens)
        # token_id -> column index within the emotion slice
        self._col = {tid: i for i, tid in enumerate(self.emotion_tokens)}
        self._mean: np.ndarray | None = None   # (n_layers, n_query)
        self._std: np.ndarray | None = None

    # ---- baseline ------------------------------------------------------------

    def fit_baseline(self, texts: list[str]) -> None:
        """Estimate per-(layer, token) logit mean/std over baseline ``texts``."""
        count = 0
        mean = None
        m2 = None  # sum of squared deviations (Welford)
        for text in texts:
            logits = self.model.residual_logit_scores(text, self.query_tokens).numpy()
            # logits: (n_layers, seq, n_query) -> treat each position as a sample
            n_layers, seq, n_query = logits.shape
            flat = logits.reshape(n_layers, seq, n_query)
            for pos in range(seq):
                x = flat[:, pos, :]  # (n_layers, n_query)
                count += 1
                if mean is None:
                    mean = np.zeros_like(x)
                    m2 = np.zeros_like(x)
                delta = x - mean
                mean = mean + delta / count
                m2 = m2 + delta * (x - mean)
        if mean is None:
            raise ValueError("No baseline positions collected.")
        self._mean = mean
        self._std = np.sqrt(m2 / max(1, count - 1)) + 1e-6

    # ---- scoring -------------------------------------------------------------

    def score_text(self, text: str, *, return_trajectory: bool = False) -> EmotionScores:
        if self._mean is None or self._std is None:
            raise RuntimeError("Call fit_baseline(...) before score_text(...).")
        logits = self.model.residual_logit_scores(text, self.query_tokens).numpy()
        z = (logits - self._mean[:, None, :]) / self._std[:, None, :]  # (L, seq, Q)

        # Regress out shared drift: subtract the mean z over random tokens at each
        # (layer, position) — a rank-1 removal of the global logit correlation the
        # paper notes (all logits rise/fall together over a conversation).
        random_z = z[:, :, self._n_emotion:].mean(axis=2, keepdims=True)  # (L, seq, 1)
        emo_z = z[:, :, : self._n_emotion] - random_z                     # (L, seq, n_emotion)

        by_emotion: dict[str, np.ndarray] = {}
        trajectory: dict[str, np.ndarray] = {}
        for emotion in EKMAN_EMOTIONS:
            cols = [self._col[t] for t in self.vocab.by_emotion[emotion] if t in self._col]
            if not cols:
                by_emotion[emotion] = np.zeros(z.shape[0])
                continue
            sel = emo_z[:, :, cols]                 # (L, seq, k)
            per_layer_pos = sel.mean(axis=2)        # (L, seq)
            by_emotion[emotion] = per_layer_pos.mean(axis=1)   # (L,)
            if return_trajectory:
                trajectory[emotion] = per_layer_pos
        return EmotionScores(
            by_emotion=by_emotion,
            trajectory=trajectory if return_trajectory else None,
        )

    def aggregate_layers(self, scores: EmotionScores, layer_lo: int = 30, layer_hi: int = 40) -> dict[str, float]:
        """Mean z over a layer band (paper aggregates over layers 30-40)."""
        return {
            e: float(v[layer_lo:layer_hi].mean()) for e, v in scores.by_emotion.items()
        }
