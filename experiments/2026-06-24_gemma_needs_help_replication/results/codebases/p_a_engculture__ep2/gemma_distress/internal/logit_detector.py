"""Logit-based internal-emotion detection (Appendix I).

Implements the paper's logit-lens emotion detector to test whether the DPO finetune
suppresses *internal* negative emotion, not just expressed emotion:

1. For each model layer, unembed the residual stream (apply the final norm + LM head =
   "logit lens"), giving a logit over the vocabulary at every position.
2. For each Ekman emotion, take the logits at that emotion's tokens and standardise each
   with its mean/std over 500 WildChat samples (z-scores). Average the z-scores over the
   category to get an emotion score per (layer, position).
3. Because the logits are globally correlated and drift over a conversation, regress out
   the common-mode signal estimated from random (non-emotion) tokens, isolating the
   emotion-specific component.

We expose conversation-level running averages (aggregated over layers 30-40, Figure 14)
and layerwise scores at three points around emotion onset (Figure 15).

Approximations vs the paper (documented in DESIGN.md): the common-mode "regress out" is
implemented as a per-layer least-squares residual of the emotion z-scores against the
random-token mean z-score across positions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import InternalConfig
from .ekman import build_emotion_tokens, sample_random_tokens

logger = logging.getLogger(__name__)


class InternalEmotionDetector:
    """Logit-lens Ekman-emotion detector over a HuggingFace causal LM."""

    def __init__(self, model, tokenizer, cfg: InternalConfig):
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.emotions = cfg.ekman_emotions

        self.emotion_tokens = build_emotion_tokens(tokenizer)
        all_emotion_ids = {t for ids in self.emotion_tokens.values() for t in ids}
        self.random_tokens = sample_random_tokens(
            tokenizer, n=max(200, len(all_emotion_ids) // 6), exclude=all_emotion_ids
        )
        # Token ids we actually need stats for.
        self._tracked = sorted(all_emotion_ids | set(self.random_tokens))
        self._tracked_index = {t: i for i, t in enumerate(self._tracked)}
        # Standardisation statistics, filled by compute_standardisation: per layer arrays.
        self._mean: Optional[np.ndarray] = None  # [n_layers, n_tracked]
        self._std: Optional[np.ndarray] = None

    # -- core logit-lens ---------------------------------------------------------

    def _tracked_logits(self, input_ids) -> np.ndarray:
        """Return logit-lens values for tracked tokens: array [n_layers, seq, n_tracked]."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple len n_layers+1, each [1, seq, hidden]
        norm = getattr(self.model, "model", self.model)
        final_norm = getattr(norm, "norm", None)
        lm_head = self.model.get_output_embeddings()

        tracked_idx = torch.tensor(self._tracked, device=input_ids.device)
        per_layer = []
        # Skip the embedding layer (index 0); use transformer block outputs 1..n.
        for hs in hidden_states[1:]:
            h = final_norm(hs) if final_norm is not None else hs
            logits = lm_head(h)  # [1, seq, vocab]
            sel = logits[0, :, tracked_idx]  # [seq, n_tracked]
            per_layer.append(sel.float().cpu().numpy())
        return np.stack(per_layer, axis=0)  # [n_layers, seq, n_tracked]

    def _encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    # -- standardisation ---------------------------------------------------------

    def compute_standardisation(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, token) mean/std over WildChat to z-score the logits."""
        n = None
        sum_: Optional[np.ndarray] = None
        sumsq: Optional[np.ndarray] = None
        count = 0
        for text in wildchat_texts[: self.cfg.standardisation_samples]:
            ids = self._encode(text)
            vals = self._tracked_logits(ids)  # [L, seq, T]
            L, seq, T = vals.shape
            if sum_ is None:
                sum_ = np.zeros((L, T))
                sumsq = np.zeros((L, T))
            sum_ += vals.sum(axis=1)
            sumsq += (vals ** 2).sum(axis=1)
            count += seq
        if count == 0 or sum_ is None:
            raise ValueError("No WildChat texts provided for standardisation.")
        self._mean = sum_ / count
        var = np.maximum(sumsq / count - self._mean ** 2, 1e-8)
        self._std = np.sqrt(var)
        logger.info("Computed standardisation over %d tokens", count)

    # -- scoring -----------------------------------------------------------------

    def _zscores(self, vals: np.ndarray) -> np.ndarray:
        """Standardise tracked logits: [L, seq, T] -> z [L, seq, T]."""
        if self._mean is None or self._std is None:
            raise RuntimeError("Call compute_standardisation() before scoring.")
        return (vals - self._mean[:, None, :]) / self._std[:, None, :]

    def _emotion_components(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Per-emotion mean z over its tokens, common-mode regressed out: [L, seq] each."""
        # Random-token common-mode per (layer, position).
        rand_cols = [self._tracked_index[t] for t in self.random_tokens]
        common = z[:, :, rand_cols].mean(axis=2)  # [L, seq]

        out: dict[str, np.ndarray] = {}
        for emotion, ids in self.emotion_tokens.items():
            cols = [self._tracked_index[t] for t in ids if t in self._tracked_index]
            if not cols:
                continue
            emo = z[:, :, cols].mean(axis=2)  # [L, seq]
            out[emotion] = self._regress_out(emo, common)
        return out

    @staticmethod
    def _regress_out(emo: np.ndarray, common: np.ndarray) -> np.ndarray:
        """Per-layer least-squares residual of emo on the common-mode across positions."""
        L, seq = emo.shape
        resid = np.empty_like(emo)
        for layer in range(L):
            x = common[layer]
            y = emo[layer]
            denom = float((x * x).sum())
            beta = float((x * y).sum() / denom) if denom > 1e-8 else 0.0
            resid[layer] = y - beta * x
        return resid

    def score_text(self, text: str) -> dict[str, np.ndarray]:
        """Return per-emotion [n_layers, seq] z-score arrays for a text."""
        ids = self._encode(text)
        z = self._zscores(self._tracked_logits(ids))
        return self._emotion_components(z)

    def conversation_level(self, text: str) -> dict[str, np.ndarray]:
        """Running-window emotion scores aggregated over the configured layers (Fig 14).

        Returns per-emotion 1-D arrays over positions (running average, window =
        ``running_window_tokens``), aggregated over ``aggregate_layers``.
        """
        comps = self.score_text(text)
        lo, hi = self.cfg.aggregate_layers
        win = self.cfg.running_window_tokens
        out: dict[str, np.ndarray] = {}
        for emotion, arr in comps.items():
            layer_avg = arr[lo:hi].mean(axis=0)  # [seq]
            kernel = np.ones(min(win, len(layer_avg))) / min(win, max(len(layer_avg), 1))
            out[emotion] = np.convolve(layer_avg, kernel, mode="same")
        return out

    def layerwise_summary(self, text: str) -> dict[str, np.ndarray]:
        """Per-emotion per-layer mean z over all positions (Fig 15-style summary)."""
        comps = self.score_text(text)
        return {emo: arr.mean(axis=1) for emo, arr in comps.items()}
