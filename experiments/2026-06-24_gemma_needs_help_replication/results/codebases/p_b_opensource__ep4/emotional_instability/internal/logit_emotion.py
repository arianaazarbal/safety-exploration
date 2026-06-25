"""Logit-based internal emotion detection (Appendix I, Figures 14-15).

We measure "internal" emotion by applying a logit lens to the residual stream:
at a given layer, the hidden state is passed through the model's final norm and
unembedding to obtain vocabulary logits, and we read off the logits of
emotion-describing tokens (Ekman's six categories; see emotion_lexicon.py).

Following the paper:
* each emotion-token logit is standardised (z-scored) using its mean and std
  computed over 500 WildChat samples (per layer);
* an emotion's score at a position is the average z-score over its tokens;
* because all logits rise and fall together over a conversation, we regress out
  a random-token baseline so the emotion score reflects emotion-specific
  variation rather than overall logit drift.

We take this logit-based approach (rather than trained probes) precisely because
it needs no probe-training data, as the paper notes. This module exposes a
calibrate / score API and is model-agnostic across Gemma instruct, base, and the
DPO finetune, so the vanilla-vs-DPO comparison (suppressed internal emotion)
runs by swapping the adapter. See DESIGN.md for the approximations
(final-logit-softcap omitted; surface-form token matching).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (
    EKMAN_EMOTIONS,
    INTERNAL_EMOTION_LAYERS,
    LOGIT_ZSCORE_CALIB_SAMPLES,
)
from .emotion_lexicon import build_emotion_token_ids


@dataclass
class CalibrationStats:
    # mean/std of each selected token's lens-logit, per layer: {layer: {tid: (mu, sd)}}
    mean: dict[int, np.ndarray] = field(default_factory=dict)
    std: dict[int, np.ndarray] = field(default_factory=dict)
    sel_ids: list[int] = field(default_factory=list)  # column order of the arrays.


class LogitEmotionProbe:
    def __init__(self, backend, layers=INTERNAL_EMOTION_LAYERS, n_random_tokens=500, seed=0):
        self.backend = backend            # HuggingFaceBackend (exposes .model/.tokenizer)
        self.layers = list(layers)
        self.tokenizer = backend.tokenizer
        # `fwd_model` runs the forward pass (a PeftModel applies the adapter, so
        # the DPO finetune's activations are captured). `core` is the unwrapped
        # base model, used only to read the (adapter-free) final norm and
        # unembedding weights.
        self.fwd_model = backend.model
        self.core = (
            self.fwd_model.get_base_model()
            if hasattr(self.fwd_model, "get_base_model")
            else self.fwd_model
        )
        self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)

        rng = np.random.default_rng(seed)
        emotion_all = {t for ids in self.emotion_token_ids.values() for t in ids}
        vocab_size = self.core.get_input_embeddings().weight.shape[0]
        pool = [t for t in range(vocab_size) if t not in emotion_all]
        self.random_token_ids = sorted(rng.choice(pool, size=min(n_random_tokens, len(pool)), replace=False).tolist())

        # Stable column order for all selected tokens (emotion + random).
        self.sel_ids = sorted(emotion_all | set(self.random_token_ids))
        self._col = {tid: i for i, tid in enumerate(self.sel_ids)}
        self.stats = CalibrationStats(sel_ids=self.sel_ids)

    # ------------------------------------------------------------------ #
    # Logit lens                                                          #
    # ------------------------------------------------------------------ #
    def _final_norm(self):
        # Gemma decoder final RMSNorm, read from the unwrapped base model.
        decoder = self.core.get_decoder() if hasattr(self.core, "get_decoder") else self.core.model
        return decoder.norm

    def _unembed_selected(self, hidden: "np.ndarray | object") -> np.ndarray:
        """Apply final norm + unembedding to hidden states, returning only the
        selected-token logits. `hidden` is a torch tensor [T, d]; returns [T, k].
        """
        import torch

        norm = self._final_norm()
        W = self.core.get_output_embeddings().weight  # [vocab, d]
        with torch.no_grad():
            h = norm(hidden)
            W_sel = W[self.sel_ids]                    # [k, d]
            logits = h.to(W_sel.dtype) @ W_sel.T       # [T, k]
        return logits.float().cpu().numpy()

    def _hidden_states(self, text: str):
        """Return list of per-layer hidden states [T, d] for a single text."""
        import torch

        enc = self.tokenizer(text, return_tensors="pt").to(self.fwd_model.device)
        with torch.no_grad():
            out = self.fwd_model(**enc, output_hidden_states=True)
        # hidden_states: tuple(num_layers+1) of [1, T, d]; index 0 is embeddings.
        return [out.hidden_states[layer + 1][0] for layer in range(len(out.hidden_states) - 1)]

    # ------------------------------------------------------------------ #
    # Calibration                                                         #
    # ------------------------------------------------------------------ #
    def calibrate(self, texts: list[str]):
        """Estimate per-layer per-token logit mean/std over calibration texts."""
        texts = texts[:LOGIT_ZSCORE_CALIB_SAMPLES]
        # Online accumulation of sum and sum-of-squares per layer.
        n = {layer: 0 for layer in self.layers}
        s1 = {layer: np.zeros(len(self.sel_ids)) for layer in self.layers}
        s2 = {layer: np.zeros(len(self.sel_ids)) for layer in self.layers}
        for text in texts:
            hs = self._hidden_states(text)
            for layer in self.layers:
                logits = self._unembed_selected(hs[layer])  # [T, k]
                s1[layer] += logits.sum(axis=0)
                s2[layer] += (logits ** 2).sum(axis=0)
                n[layer] += logits.shape[0]
        for layer in self.layers:
            mu = s1[layer] / max(1, n[layer])
            var = s2[layer] / max(1, n[layer]) - mu ** 2
            self.stats.mean[layer] = mu
            self.stats.std[layer] = np.sqrt(np.clip(var, 1e-8, None))
        return self.stats

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #
    def _emotion_cols(self, emotion: str) -> list[int]:
        return [self._col[t] for t in self.emotion_token_ids[emotion]]

    def _random_cols(self) -> list[int]:
        return [self._col[t] for t in self.random_token_ids]

    def score_text(self, text: str, regress_random: bool = True) -> dict:
        """Return per-layer per-emotion z-score trajectories for one text.

        Output: {layer: {"tokens": T, emotion: np.ndarray[T], ...}}. When
        `regress_random` is set, the random-token baseline is regressed out of
        each emotion series (residual of OLS on the random-token mean z).
        """
        hs = self._hidden_states(text)
        result: dict[int, dict] = {}
        for layer in self.layers:
            logits = self._unembed_selected(hs[layer])           # [T, k]
            z = (logits - self.stats.mean[layer]) / self.stats.std[layer]
            rand_z = z[:, self._random_cols()].mean(axis=1)      # [T]
            layer_out = {"tokens": z.shape[0]}
            for emotion in EKMAN_EMOTIONS:
                cols = self._emotion_cols(emotion)
                if not cols:
                    layer_out[emotion] = np.zeros(z.shape[0])
                    continue
                e_z = z[:, cols].mean(axis=1)                    # [T]
                if regress_random and z.shape[0] >= 2:
                    e_z = _regress_out(e_z, rand_z)
                layer_out[emotion] = e_z
            result[layer] = layer_out
        return result

    def conversation_emotion(
        self, text: str, layers_to_aggregate=None, window: int = 400
    ) -> dict:
        """Running-average emotion z-scores over a conversation, aggregated over
        layers (Figure 14 plots layers 30-40, windowed over 400 tokens).
        """
        layers_to_aggregate = layers_to_aggregate or self.layers
        per_layer = self.score_text(text)
        out = {}
        for emotion in EKMAN_EMOTIONS:
            stacked = np.stack([per_layer[l][emotion] for l in layers_to_aggregate])
            series = stacked.mean(axis=0)                        # [T]
            out[emotion] = _running_mean(series, window)
        return out


def _regress_out(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return residual of OLS regression of y on x (with intercept)."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _running_mean(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) <= 1:
        return series
    k = min(window, len(series))
    kernel = np.ones(k) / k
    return np.convolve(series, kernel, mode="same")
