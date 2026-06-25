"""Logit-based internal emotion detection (Appendix I).

Method (Appendix I, verbatim summary):
  * Classify the vocabulary into Ekman's 6 emotions (emotion_lexicon.py).
  * For each layer, unembed the residual stream (apply the final norm + LM head
    to the hidden state at that layer) to get logits over the vocabulary.
  * Standardise each token logit by its mean/std computed over 500 WildChat
    samples (a per-token z-score).
  * The emotion score at a layer/position = mean z-score over that emotion's
    tokens.
  * Because all logits are correlated and drift over a conversation, regress out
    the mean z-score of a set of random control tokens, leaving an emotion
    score per layer per position.

Conversation-level reporting aggregates over layers 30-40 and plots a running
average over 400-token windows (Figure 14). Layerwise reporting averages over
tokens at three points around emotion onset (Figure 15).

This module needs the residual stream, so it always uses the transformers
backend (HFLocalClient), not vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import DATA_DIR
from .emotion_lexicon import EKMAN, build_token_emotion_map

CALIBRATION_CACHE = DATA_DIR / "logit_calibration.npz"
LAYER_BAND = (30, 40)  # default aggregation band (Figure 14)


@dataclass
class Calibration:
    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]


class EmotionProbe:
    """Wraps a transformers Gemma model to compute per-layer emotion z-scores."""

    def __init__(self, hf_client, *, n_control_tokens: int = 500):
        # hf_client must be an HFLocalClient (exposes .model / .tokenizer).
        self.client = hf_client
        self.n_control = n_control_tokens
        self._tok_emotion: dict[int, str] | None = None
        self._control_ids: Optional[np.ndarray] = None
        self._calib: Optional[Calibration] = None

    # ------------------------------------------------------------------ #
    @property
    def tokenizer(self):
        return self.client.tokenizer

    @property
    def model(self):
        return self.client.model

    def _ensure_lexicon(self):
        if self._tok_emotion is None:
            self._tok_emotion = build_token_emotion_map(self.tokenizer)
            import random
            rng = random.Random(0)
            vocab_size = len(self.tokenizer)
            emo_ids = set(self._tok_emotion)
            controls = [i for i in range(vocab_size) if i not in emo_ids]
            self._control_ids = np.array(rng.sample(controls, self.n_control))

    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str) -> np.ndarray:
        """Return logits per layer for every token position: [n_layers, seq, vocab].

        We unembed each layer's hidden state through the model's final norm and
        LM head (the "logit lens").
        """
        import torch
        tok = self.tokenizer
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple(n_layers+1) of [1, seq, d]
        norm = _final_norm(self.model)
        head = self.model.get_output_embeddings()
        logits_per_layer = []
        with torch.no_grad():
            for h in hidden[1:]:  # skip embedding layer
                normed = norm(h) if norm is not None else h
                lg = head(normed)  # [1, seq, vocab]
                logits_per_layer.append(lg[0].float().cpu().numpy())
        return np.stack(logits_per_layer, axis=0)

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str], *, force: bool = False):
        """Compute per-layer/per-token mean and std over WildChat samples."""
        if CALIBRATION_CACHE.exists() and not force:
            d = np.load(CALIBRATION_CACHE)
            self._calib = Calibration(d["mean"], d["std"])
            return self._calib
        self._ensure_lexicon()
        sums = sumsq = None
        count = 0
        for text in wildchat_texts[:self.n_control_samples()]:
            lg = self._layer_logits(text)            # [L, seq, V]
            flat = lg.reshape(lg.shape[0], -1, lg.shape[2])  # [L, seq, V]
            s = flat.sum(axis=1)                     # [L, V]
            sq = (flat ** 2).sum(axis=1)
            n = flat.shape[1]
            sums = s if sums is None else sums + s
            sumsq = sq if sumsq is None else sumsq + sq
            count += n
        mean = sums / count
        var = np.maximum(sumsq / count - mean ** 2, 1e-6)
        self._calib = Calibration(mean, np.sqrt(var))
        np.savez(CALIBRATION_CACHE, mean=self._calib.mean, std=self._calib.std)
        return self._calib

    def n_control_samples(self) -> int:
        return 500  # paper: standardise over 500 WildChat samples

    # ------------------------------------------------------------------ #
    def emotion_scores(self, text: str) -> dict[str, np.ndarray]:
        """Per-layer emotion z-scores for `text`, regressing out control drift.

        Returns {emotion: array[n_layers]} averaged over token positions.
        """
        assert self._calib is not None, "call calibrate() first"
        self._ensure_lexicon()
        lg = self._layer_logits(text)               # [L, seq, V]
        z = (lg - self._calib.mean[:, None, :]) / self._calib.std[:, None, :]

        # Control drift = mean z over random control tokens, per layer/position.
        control = z[:, :, self._control_ids].mean(axis=2)   # [L, seq]

        scores: dict[str, np.ndarray] = {}
        emo_to_ids: dict[str, list[int]] = {e: [] for e in EKMAN}
        for tid, emo in self._tok_emotion.items():
            emo_to_ids[emo].append(tid)
        for emo, ids in emo_to_ids.items():
            if not ids:
                scores[emo] = np.full(lg.shape[0], np.nan)
                continue
            emo_z = z[:, :, ids].mean(axis=2)               # [L, seq]
            residual = emo_z - control                      # regress out drift
            scores[emo] = residual.mean(axis=1)             # avg over positions
        return scores

    def conversation_trajectory(self, text: str, *, window: int = 400,
                                band: tuple[int, int] = LAYER_BAND
                                ) -> dict[str, np.ndarray]:
        """Running-average emotion score over the conversation (Figure 14).

        Aggregates layers in `band`, then computes a running mean over `window`
        token positions.
        """
        assert self._calib is not None, "call calibrate() first"
        self._ensure_lexicon()
        lg = self._layer_logits(text)
        z = (lg - self._calib.mean[:, None, :]) / self._calib.std[:, None, :]
        lo, hi = band
        z = z[lo:hi]                                        # [band, seq, V]
        control = z[:, :, self._control_ids].mean(axis=2).mean(axis=0)  # [seq]
        out = {}
        emo_to_ids: dict[str, list[int]] = {e: [] for e in EKMAN}
        for tid, emo in self._tok_emotion.items():
            emo_to_ids[emo].append(tid)
        for emo, ids in emo_to_ids.items():
            if not ids:
                continue
            emo_z = z[:, :, ids].mean(axis=2).mean(axis=0)  # [seq]
            residual = emo_z - control
            out[emo] = _running_mean(residual, window)
        return out


def _final_norm(model):
    """Best-effort fetch of the model's pre-LM-head norm."""
    for attr in ("model",):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "norm"):
            return inner.norm
    return getattr(getattr(model, "model", model), "norm", None)


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return np.array([x.mean()]) if len(x) else x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")
