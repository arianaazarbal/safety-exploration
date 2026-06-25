"""Logit-lens internal-emotion detection (Appendix I).

Method (paper): unembed the residual stream and standardise each emotion-token
logit by its mean/std over 500 WildChat samples; average the z-scores over the
tokens in each Ekman category to get a per-layer, per-position emotion score.
Because all logits drift together over a conversation, we regress out a "drift"
signal estimated from random non-emotion tokens.

We avoid materialising the full ~256k-vocab logits: only the selected emotion +
random unembedding rows are multiplied in (``_selected_logits``). This is the
``logit-based approach`` the paper chose over trained probes ("avoids generating
probe data").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.hf_local import HFLocalClient
from .emotion_tokens import EmotionTokenSets


@dataclass
class EmotionTrajectory:
    emotions: list[str]
    # per-emotion arrays over positions, aggregated across the report layer window
    scores: dict[str, np.ndarray]
    token_ids: list[int]


class EmotionProbe:
    def __init__(self, client: HFLocalClient, token_sets: EmotionTokenSets,
                 aggregate_layers: tuple[int, int] = (30, 40)):
        self.client = client
        self.ts = token_sets
        self.lo, self.hi = aggregate_layers
        self.emotions = list(token_sets.by_emotion.keys())

        # Selected unembedding rows (emotion tokens first, then random).
        self._sel_ids = token_sets.all_emotion_ids + token_sets.random_ids
        self._id_to_col = {tid: i for i, tid in enumerate(self._sel_ids)}
        base = client.model.get_base_model() if hasattr(client.model, "get_base_model") else client.model
        self._norm = base.model.norm
        W = base.get_output_embeddings().weight              # (V, d)
        self._W_sel = W[self._sel_ids].detach()              # (n_sel, d)

        self.mean_: torch.Tensor | None = None               # (L+1, n_sel)
        self.std_: torch.Tensor | None = None

    @torch.no_grad()
    def _selected_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (L+1, T, d) -> selected logits (L+1, T, n_sel)."""
        normed = self._norm(hidden.to(self._W_sel.dtype))
        return normed @ self._W_sel.T

    # ---- baseline standardisation (500 WildChat samples) -----------------
    @torch.no_grad()
    def fit(self, baseline_texts: list[str]) -> "EmotionProbe":
        sums = sqsums = counts = None
        for text in baseline_texts:
            hidden, _ = self.client.residual_streams(text)            # (L+1, T, d)
            logits = self._selected_logits(hidden).float()            # (L+1, T, n_sel)
            s = logits.sum(dim=1).cpu()                               # (L+1, n_sel)
            sq = (logits ** 2).sum(dim=1).cpu()
            t = logits.shape[1]
            if sums is None:
                sums, sqsums, counts = s, sq, t
            else:
                sums += s; sqsums += sq; counts += t
        self.mean_ = sums / counts
        self.std_ = (sqsums / counts - self.mean_ ** 2).clamp_min(1e-6).sqrt()
        return self

    # ---- scoring ---------------------------------------------------------
    def _emotion_cols(self, emotion: str) -> list[int]:
        return [self._id_to_col[t] for t in self.ts.by_emotion[emotion]]

    @property
    def _random_cols(self) -> list[int]:
        return [self._id_to_col[t] for t in self.ts.random_ids]

    @torch.no_grad()
    def score(self, text: str) -> EmotionTrajectory:
        assert self.mean_ is not None, "call fit() first"
        hidden, ids = self.client.residual_streams(text)
        z = ((self._selected_logits(hidden).float().cpu() - self.mean_) / self.std_)  # (L+1,T,n_sel)

        # Aggregate over report layer window.
        zl = z[self.lo : self.hi].mean(dim=0).numpy()                # (T, n_sel)
        drift = zl[:, self._random_cols].mean(axis=1)                # (T,)

        out: dict[str, np.ndarray] = {}
        for emo in self.emotions:
            raw = zl[:, self._emotion_cols(emo)].mean(axis=1)        # (T,)
            out[emo] = _regress_out(raw, drift)
        return EmotionTrajectory(emotions=self.emotions, scores=out, token_ids=ids)


def _regress_out(signal: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Residual of ``signal`` after regressing on ``drift`` (removes global drift)."""
    if np.std(drift) < 1e-8:
        return signal
    beta = np.cov(signal, drift)[0, 1] / np.var(drift)
    return signal - beta * drift


def running_average(series: np.ndarray, token_ids: list[int], window_tokens: int = 400,
                    tokenizer=None) -> np.ndarray:
    """Smooth a per-token series with a trailing window (paper plots 400-token windows)."""
    if window_tokens <= 1:
        return series
    kernel = np.ones(min(window_tokens, len(series))) / min(window_tokens, len(series))
    return np.convolve(series, kernel, mode="same")
