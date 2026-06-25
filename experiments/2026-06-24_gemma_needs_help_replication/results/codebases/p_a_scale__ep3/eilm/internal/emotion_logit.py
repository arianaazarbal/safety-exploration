"""Logit-based internal emotion detection (Appendix I).

Method (from the paper):
  * Classify the vocabulary into Ekman's six emotions (anger, surprise, disgust,
    joy, fear, sadness) -> ~1200 emotion tokens (~200 each); see data/ekman.py.
  * For an emotion at a given layer, *unembed the residual stream* (apply the
    final norm + LM head to that layer's hidden state) to get logits, standardise
    each emotion-token logit with its mean/std over 500 WildChat samples
    (z-score), and average the z-scores over the emotion's tokens.
  * Because all logits rise/fall together over a conversation, regress out the
    correlation with random tokens to isolate emotion-specific signal.
  * Aggregate over layers 30-40 for conversation-level scores; running average
    over 400-token windows.

This requires hidden states / the LM head, so it uses the transformers backend
(HFModel), not vLLM. Runs on Gemma only (within scope).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..data.ekman import build_emotion_token_ids

logger = logging.getLogger("eilm.internal")


@dataclass
class CalibStats:
    mean: np.ndarray   # [num_layers, n_tracked]
    std: np.ndarray    # [num_layers, n_tracked]
    token_ids: List[int]
    emotion_slices: Dict[str, slice]
    random_slice: slice


class EmotionProbe:
    def __init__(self, hf_model, emotions: List[str], tokens_per_emotion: int = 200,
                 n_random: int = 200, seed: int = 0):
        import torch

        self.torch = torch
        self.model = hf_model.model
        self.tokenizer = hf_model.tokenizer
        self.emotions = emotions
        self.device = next(self.model.parameters()).device

        # Locate the final norm and LM head for unembedding intermediate layers.
        self._norm = self._find_final_norm()
        self._lm_head = self.model.get_output_embeddings()
        self._unembed_w = self._lm_head.weight  # [vocab, hidden]

        # Build tracked token id table: emotion tokens then random tokens.
        by_emotion = build_emotion_token_ids(self.tokenizer, emotions, tokens_per_emotion)
        token_ids: List[int] = []
        self.emotion_slices: Dict[str, slice] = {}
        for e in emotions:
            ids = by_emotion[e]
            start = len(token_ids)
            token_ids.extend(ids)
            self.emotion_slices[e] = slice(start, len(token_ids))
        emotion_set = set(token_ids)

        rng = np.random.default_rng(seed)
        vocab_size = self._lm_head.weight.shape[0]
        rand_ids = []
        while len(rand_ids) < n_random:
            cand = int(rng.integers(0, vocab_size))
            if cand not in emotion_set:
                rand_ids.append(cand)
                emotion_set.add(cand)
        rstart = len(token_ids)
        token_ids.extend(rand_ids)
        self.random_slice = slice(rstart, len(token_ids))
        self.token_ids = token_ids
        self._token_ids_t = torch.tensor(token_ids, device=self.device)
        # Pre-slice the unembedding rows for the tracked tokens only, so we never
        # materialise full-vocab logits (which would OOM at long sequence lengths).
        self._w_sub = self._unembed_w.index_select(0, self._token_ids_t).detach()  # [n_tracked, hidden]
        self.stats: Optional[CalibStats] = None

    def _find_final_norm(self):
        # Gemma3: model.model.norm (RMSNorm). Fall back to common attribute names.
        for attr in ("norm", "final_layernorm"):
            base = getattr(self.model, "model", self.model)
            if hasattr(base, attr):
                return getattr(base, attr)
        raise AttributeError("Could not locate final norm layer for unembedding")

    def _tracked_logits(self, text: str) -> np.ndarray:
        """Return tracked-token logits per layer for every position.
        Shape: [num_layers, seq_len, n_tracked]."""
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=12000).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple len num_layers+1 (embeddings + each layer)
        hs = out.hidden_states[1:]  # drop embedding layer
        w_sub = self._w_sub.to(hs[0].dtype)
        per_layer = []
        for h in hs:
            normed = self._norm(h)[0]                  # [seq, hidden]
            # Project onto tracked-token unembedding rows only: [seq, n_tracked].
            tracked = torch.matmul(normed, w_sub.t())
            per_layer.append(tracked.float().cpu().numpy())
        return np.stack(per_layer, axis=0)             # [layers, seq, n_tracked]

    # --- calibration -------------------------------------------------------
    def calibrate(self, wildchat_texts: List[str]) -> CalibStats:
        """Estimate per-(layer, tracked-token) mean/std over WildChat positions."""
        sums = None
        sqsums = None
        counts = 0
        for text in wildchat_texts:
            tl = self._tracked_logits(text)            # [L, S, T]
            flat = tl.reshape(tl.shape[0], -1, tl.shape[2])  # [L, S, T]
            s = flat.sum(axis=1)                       # [L, T]
            sq = (flat ** 2).sum(axis=1)
            n = flat.shape[1]
            sums = s if sums is None else sums + s
            sqsums = sq if sqsums is None else sqsums + sq
            counts += n
        mean = sums / counts
        var = np.maximum(sqsums / counts - mean ** 2, 1e-6)
        self.stats = CalibStats(
            mean=mean, std=np.sqrt(var), token_ids=self.token_ids,
            emotion_slices=self.emotion_slices, random_slice=self.random_slice,
        )
        return self.stats

    # --- scoring -----------------------------------------------------------
    def score_text(self, text: str) -> Dict[str, np.ndarray]:
        """Per-emotion z-score arrays, shape [num_layers, seq_len], with the
        random-token baseline regressed out (subtracted)."""
        if self.stats is None:
            raise RuntimeError("Probe not calibrated; call calibrate() first")
        tl = self._tracked_logits(text)                # [L, S, T]
        z = (tl - self.stats.mean[:, None, :]) / self.stats.std[:, None, :]
        random_baseline = z[:, :, self.random_slice].mean(axis=2)  # [L, S]
        scores = {}
        for e, sl in self.emotion_slices.items():
            emo_z = z[:, :, sl].mean(axis=2)           # [L, S]
            scores[e] = emo_z - random_baseline
        return scores

    def conversation_scores(self, text: str, layers=(30, 40),
                            window: int = 400) -> Dict[str, np.ndarray]:
        """Conversation-level scores: average over `layers` window, then a running
        average over `window` tokens (Figure 14)."""
        per_layer = self.score_text(text)
        lo, hi = layers
        out = {}
        for e, arr in per_layer.items():
            layer_avg = arr[lo:hi].mean(axis=0)        # [S]
            out[e] = _running_mean(layer_avg, window)
        return out


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    w = min(window, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")
