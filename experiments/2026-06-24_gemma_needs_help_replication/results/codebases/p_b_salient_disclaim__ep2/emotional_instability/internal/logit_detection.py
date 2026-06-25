"""Logit-based internal emotion detection (Appendix I).

Method (Appendix I): to score an emotion at a given layer and conversation
position, unembed the residual stream (project the hidden state through the
output embedding / LM head), standardise each logit against its mean and std
over 500 WildChat samples (a per-token-id z-score), then average these z-scores
over the tokens in the emotion category. Because all logits are correlated and
drift over a conversation, we additionally regress out the common-mode signal
estimated from random tokens, yielding a drift-corrected emotion score at each
layer and position.

Outputs support:
  - Figure 14: conversation-level trajectory, aggregated over layers 30-40,
    plotted as a running average over 400-token windows.
  - Figure 15: layerwise scores at three conversation stages (40-20 tokens
    before emotion onset, 0-20 tokens before, and the final 20 tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config.settings import SETTINGS
from .emotion_tokens import EKMAN_EMOTIONS, build_emotion_token_ids


@dataclass
class CalibrationStats:
    # Per layer: mean/std of each tracked token id's logit over WildChat positions.
    emotion_token_ids: dict[str, list[int]]
    random_token_ids: list[int]
    mean: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> [n_tracked]
    std: dict[int, np.ndarray] = field(default_factory=dict)
    tracked_ids: list[int] = field(default_factory=list)       # order of columns


class InternalEmotionDetector:
    """Wraps a local HF Gemma model to compute logit-based internal emotion scores.

    Both the vanilla instruct model and the DPO finetune are run through the same
    detector so their internal emotion trajectories can be compared (Figure 14/15).
    """

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        n_random_tokens: int = 500,
        seed: int = SETTINGS.seed,
    ):
        self.model_id = model_id
        self._dtype = dtype
        self._device_map = device_map
        self._n_random = n_random_tokens
        self._seed = seed
        self._model = None
        self._tokenizer = None
        self.calib: Optional[CalibrationStats] = None

    # ------------------------------------------------------------------ #
    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, self._dtype),
                device_map=self._device_map,
                output_hidden_states=True,
            )
            self._model.eval()
        return self._model, self._tokenizer

    def _lm_head(self):
        # Project residual stream -> vocab logits via the output embedding.
        return self._model.get_output_embeddings()

    def _hidden_logits_for_ids(self, text: str, tracked_ids: list[int]) -> np.ndarray:
        """Return logits for `tracked_ids` at every (layer, position).

        Shape: [n_layers, seq_len, n_tracked].
        """
        import torch

        model, tok = self._ensure_model()
        head = self._lm_head()
        W = head.weight  # [vocab, hidden]
        cols = W[tracked_ids, :]  # [n_tracked, hidden]

        enc = tok(text, return_tensors="pt", truncation=True, max_length=12000)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # hidden_states: tuple(len = n_layers+1) of [1, seq, hidden]
        hs = out.hidden_states
        per_layer = []
        for layer_h in hs[1:]:  # skip embedding layer
            h = layer_h[0]  # [seq, hidden]
            logits = h.to(cols.dtype) @ cols.t()  # [seq, n_tracked]
            per_layer.append(logits.float().cpu().numpy())
        return np.stack(per_layer, axis=0)  # [n_layers, seq, n_tracked]

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str]) -> CalibrationStats:
        """Estimate per-(layer, token) mean/std of logits over WildChat positions."""
        model, tok = self._ensure_model()
        emotion_ids = build_emotion_token_ids(tok, target_total=SETTINGS.internal_emotion_token_target)

        rng = np.random.default_rng(self._seed)
        vocab_size = model.get_output_embeddings().weight.shape[0]
        random_ids = sorted(rng.choice(vocab_size, size=self._n_random, replace=False).tolist())

        tracked = sorted({tid for ids in emotion_ids.values() for tid in ids} | set(random_ids))
        idx_of = {tid: i for i, tid in enumerate(tracked)}

        # Accumulate running sum / sumsq per layer over all positions.
        sums: dict[int, np.ndarray] = {}
        sqs: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        for text in wildchat_texts:
            arr = self._hidden_logits_for_ids(text, tracked)  # [L, S, T]
            L, S, T = arr.shape
            for layer in range(L):
                a = arr[layer]  # [S, T]
                sums.setdefault(layer, np.zeros(T))
                sqs.setdefault(layer, np.zeros(T))
                counts.setdefault(layer, 0)
                sums[layer] += a.sum(axis=0)
                sqs[layer] += (a ** 2).sum(axis=0)
                counts[layer] += S

        mean, std = {}, {}
        for layer in sums:
            n = max(1, counts[layer])
            m = sums[layer] / n
            var = np.maximum(sqs[layer] / n - m ** 2, 1e-6)
            mean[layer] = m
            std[layer] = np.sqrt(var)

        self.calib = CalibrationStats(
            emotion_token_ids=emotion_ids,
            random_token_ids=random_ids,
            mean=mean,
            std=std,
            tracked_ids=tracked,
        )
        return self.calib

    # ------------------------------------------------------------------ #
    def score_conversation(self, text: str) -> dict:
        """Return drift-corrected emotion z-scores per layer & position.

        Output: {emotion: ndarray[n_layers, seq_len]} plus 'seq_len'.
        """
        assert self.calib is not None, "call calibrate() first"
        calib = self.calib
        idx_of = {tid: i for i, tid in enumerate(calib.tracked_ids)}

        arr = self._hidden_logits_for_ids(text, calib.tracked_ids)  # [L, S, T]
        L, S, T = arr.shape

        rand_cols = [idx_of[t] for t in calib.random_token_ids]

        scores: dict[str, np.ndarray] = {e: np.zeros((L, S)) for e in EKMAN_EMOTIONS}
        for layer in range(L):
            m = calib.mean.get(layer)
            sd = calib.std.get(layer)
            if m is None:
                continue
            z = (arr[layer] - m) / sd  # [S, T] z-scored logits
            # Common-mode drift signal from random tokens, per position.
            drift = z[:, rand_cols].mean(axis=1)  # [S]
            for emotion, ids in calib.emotion_token_ids.items():
                cols = [idx_of[t] for t in ids if t in idx_of]
                if not cols:
                    continue
                raw = z[:, cols].mean(axis=1)  # [S]
                # Regress out drift: residual of raw ~ drift across positions.
                scores[emotion][layer] = _regress_out(raw, drift)
        scores["seq_len"] = S
        return scores

    # ------------------------------------------------------------------ #
    @staticmethod
    def conversation_trajectory(
        scores: dict, layers: tuple[int, int] = SETTINGS.internal_layers_for_conv_plot, window: int = 400
    ) -> dict[str, np.ndarray]:
        """Figure 14: per-emotion running average over `window` positions,
        aggregated over `layers` (start, end_exclusive)."""
        lo, hi = layers
        out = {}
        for emotion in EKMAN_EMOTIONS:
            arr = scores[emotion][lo:hi].mean(axis=0)  # [S]
            out[emotion] = _running_average(arr, window)
        return out


def _regress_out(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return residual of y after removing its linear dependence on x (per layer)."""
    if x.std() < 1e-8:
        return y - y.mean()
    beta = np.cov(y, x, bias=True)[0, 1] / (x.var() + 1e-8)
    return y - beta * x


def _running_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) <= 1:
        return arr
    kernel = np.ones(min(window, len(arr))) / min(window, len(arr))
    return np.convolve(arr, kernel, mode="same")
