"""Logit-based internal-emotion detection (App. I).

Method (per the paper):
  1. Classify the Gemma dictionary into Ekman emotions (~200 tokens each) -> :mod:`.emotion_lexicon`.
  2. Unembed the residual stream at each layer into vocab logits, but only for the emotion
     tokens (+ a random control set) -- restricting columns keeps this tractable on the full
     256k-token Gemma vocab.
  3. Standardise each selected logit with its mean/std over 500 WildChat samples (per layer).
  4. Average the z-scores over each emotion's tokens to get an emotion score at every layer and
     token position.
  5. Because all logits are globally correlated and drift over a conversation, regress out a
     random-token control signal per layer, leaving a corrected emotion score per position.
  6. Aggregate over layers 30-40 and take a running average over 400-token windows for the
     conversation-level trajectory (Figure 14), or average over a window for the staged
     before/at-onset/end comparison (Figure 15).

This module computes the scores; :mod:`.run_probe` orchestrates the vanilla-vs-DPO comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ProbeConfig
from ..models.hf_backend import HFBackend
from ..utils import Message
from .emotion_lexicon import build_emotion_lexicon


def _unwrap(model):
    """Return the underlying CausalLM, unwrapping a PEFT adapter if present."""
    m = model
    if hasattr(m, "base_model"):
        m = m.base_model
        if hasattr(m, "model"):
            m = m.model
    return m


@dataclass
class _Readout:
    norm: object        # final RMSNorm module
    weight: object      # output embedding weight [vocab, d_model] (torch tensor)


class EmotionProbe:
    def __init__(self, backend: HFBackend, *, cfg: ProbeConfig | None = None, seed: int = 0):
        if not isinstance(backend, HFBackend):
            raise TypeError("EmotionProbe requires a local HFBackend (needs residual stream).")
        self.backend = backend
        self.cfg = cfg or ProbeConfig()
        self.seed = seed
        self.lexicon: dict[str, list[int]] = {}
        self.selected_ids: list[int] = []
        self.emotion_cols: dict[str, list[int]] = {}   # indices into selected_ids
        self.control_cols: list[int] = []
        self._mean: np.ndarray | None = None           # [num_layers, k]
        self._std: np.ndarray | None = None

    # ---- setup -----------------------------------------------------------------------
    @property
    def _torch(self):
        return self.backend._torch

    def _readout(self) -> _Readout:
        inner = _unwrap(self.backend.model)
        return _Readout(norm=inner.model.norm, weight=inner.get_output_embeddings().weight)

    def build_lexicon(self, method: str = "seed", llm_backend=None) -> dict[str, list[int]]:
        self.lexicon = build_emotion_lexicon(
            self.backend.tokenizer, self.cfg.ekman_emotions, method=method, llm_backend=llm_backend,
        )
        rng = np.random.default_rng(self.seed)
        vocab_size = self.backend.tokenizer.vocab_size
        emotion_ids = sorted({t for ids in self.lexicon.values() for t in ids})
        emotion_set = set(emotion_ids)
        # Random control tokens disjoint from the emotion sets.
        control_ids = []
        while len(control_ids) < self.cfg.n_random_control_tokens:
            cand = int(rng.integers(0, vocab_size))
            if cand not in emotion_set:
                control_ids.append(cand)
                emotion_set.add(cand)
        self.selected_ids = emotion_ids + control_ids
        pos = {tid: i for i, tid in enumerate(self.selected_ids)}
        self.emotion_cols = {e: [pos[t] for t in ids] for e, ids in self.lexicon.items()}
        self.control_cols = [pos[t] for t in control_ids]
        return self.lexicon

    # ---- forward / readout -----------------------------------------------------------
    def _selected_logits(self, messages: list[Message], prefix: str | None = None) -> np.ndarray:
        """Return [num_layers, seq, k] selected-token logits across all hidden layers."""
        torch = self._torch
        readout = self._readout()
        hidden, _ids = self.backend.residual_stream(messages, prefix=prefix)
        w_sel = readout.weight[self.selected_ids]  # [k, d]
        out = []
        for h in hidden:  # each [seq, d]
            with torch.no_grad():
                normed = readout.norm(h)
                logits = normed.to(w_sel.dtype) @ w_sel.t()  # [seq, k]
            out.append(logits.float().cpu().numpy())
        return np.stack(out, axis=0)

    def fit_baseline(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, selected-token) mean/std over WildChat token positions."""
        n = self.cfg.n_standardisation_samples
        texts = wildchat_texts[:n]
        sum_ = None
        sumsq = None
        count = 0
        for text in texts:
            logits = self._selected_logits([{"role": "user", "content": text}])  # [L, seq, k]
            L, seq, k = logits.shape
            flat = logits.reshape(L, seq, k)
            if sum_ is None:
                sum_ = np.zeros((L, k))
                sumsq = np.zeros((L, k))
            sum_ += flat.sum(axis=1)
            sumsq += (flat ** 2).sum(axis=1)
            count += seq
        if count == 0:
            raise ValueError("No WildChat baseline tokens collected.")
        mean = sum_ / count
        var = np.maximum(sumsq / count - mean ** 2, 1e-8)
        self._mean = mean
        self._std = np.sqrt(var)

    # ---- scoring ---------------------------------------------------------------------
    def _zscores(self, logits: np.ndarray) -> np.ndarray:
        """Standardise [L, seq, k] logits with the fitted baseline -> z-scores."""
        if self._mean is None:
            raise RuntimeError("Call fit_baseline before scoring.")
        return (logits - self._mean[:, None, :]) / self._std[:, None, :]

    def _emotion_layer_scores(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Per emotion: [L, seq] mean z over the emotion's columns, with control regressed out."""
        control = z[:, :, self.control_cols].mean(axis=2)  # [L, seq]
        result: dict[str, np.ndarray] = {}
        for emotion, cols in self.emotion_cols.items():
            if not cols:
                continue
            raw = z[:, :, cols].mean(axis=2)  # [L, seq]
            corrected = np.empty_like(raw)
            for layer in range(raw.shape[0]):
                corrected[layer] = _regress_out(raw[layer], control[layer])
            result[emotion] = corrected
        return result

    def score_conversation(self, messages: list[Message], prefix: str | None = None) -> dict:
        """Return corrected emotion scores for a conversation.

        Output:
          per_emotion_layer: {emotion: [L, seq]} corrected z-scores
          trajectory:        {emotion: [seq]} aggregated over ``aggregate_layers`` + running avg
          seq_len:           int
        """
        logits = self._selected_logits(messages, prefix=prefix)
        z = self._zscores(logits)
        layer_scores = self._emotion_layer_scores(z)
        lo, hi = self.cfg.aggregate_layers
        trajectory = {}
        for emotion, arr in layer_scores.items():
            agg = arr[lo:hi].mean(axis=0)  # average target layers -> [seq]
            trajectory[emotion] = _running_average(agg, self.cfg.running_window_tokens)
        return {
            "per_emotion_layer": layer_scores,
            "trajectory": trajectory,
            "seq_len": logits.shape[1],
        }

    def staged_scores(self, messages: list[Message], onset_pos: int) -> dict:
        """Figure-15-style staged averages around an onset position.

        Returns mean corrected score over three windows: 20-40 tokens before onset,
        0-20 before onset, and the final 20 tokens, aggregated over ``aggregate_layers``.
        """
        out = self.score_conversation(messages)
        lo, hi = self.cfg.aggregate_layers
        stages = {}
        for emotion, arr in out["per_emotion_layer"].items():
            agg = arr[lo:hi].mean(axis=0)
            seq = agg.shape[0]
            o = min(max(onset_pos, 0), seq)
            stages[emotion] = {
                "pre_40_20": _safe_mean(agg[max(0, o - 40):max(0, o - 20)]),
                "pre_20_0": _safe_mean(agg[max(0, o - 20):o]),
                "final_20": _safe_mean(agg[max(0, seq - 20):seq]),
            }
        return stages


def _safe_mean(window: np.ndarray) -> float | None:
    return float(np.mean(window)) if window.size else None


def _regress_out(signal: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Return residual of ``signal`` after OLS regression on ``control`` (+ intercept)."""
    if signal.size < 2:
        return signal
    X = np.column_stack([np.ones_like(control), control])
    coef, *_ = np.linalg.lstsq(X, signal, rcond=None)
    return signal - X @ coef


def _running_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size <= 1:
        return x
    window = min(window, x.size)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")
