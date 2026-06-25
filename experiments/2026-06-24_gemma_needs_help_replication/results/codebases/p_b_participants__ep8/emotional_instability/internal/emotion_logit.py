"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I, paraphrased):
  * Classify Gemma vocab tokens into Ekman's 6 emotions (~1200 tokens).
  * To score an emotion at a given position/layer: unembed the residual stream,
    standardise each token's logit by its mean/std over 500 WildChat samples
    (z-score), then average those z-scores over the emotion's token set.
  * Because all logits are correlated and drift over a conversation, regress out
    the correlation with a set of random control tokens to isolate the emotion
    signal at each layer / conversation position.

This lets us check whether the DPO model has *lower internal* negative emotion
(not merely suppressed expression). The paper aggregates over layers 30-40 for
conversation-level plots (Figure 14) and probes per-layer around emotion onset
(Figure 15).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .emotion_lexicon import EKMAN_EMOTIONS, build_emotion_token_ids


@dataclass
class EmotionScores:
    # emotion -> per-layer z-score (length n_layers+1)
    per_layer: dict[str, list[float]]
    layers_aggregated: dict[str, float]   # mean over a layer window (e.g. 30-40)


class EmotionLogitProbe:
    def __init__(self, gemma_client, *, n_random_controls: int = 200,
                 seed: int = 0) -> None:
        """``gemma_client`` is a ``GemmaHFClient`` (needs hidden_states/unembed)."""
        self.client = gemma_client
        self.tok = gemma_client.tokenizer
        self.emotion_ids = build_emotion_token_ids(self.tok)
        rng = random.Random(seed)
        vocab_size = len(self.tok)
        self.control_ids = rng.sample(range(vocab_size), n_random_controls)
        self._mean = None   # [vocab] baseline mean logit
        self._std = None    # [vocab] baseline std logit

    # -- baseline standardisation over WildChat -----------------------------

    def calibrate(self, wildchat_texts: list[str], *, layer: int = -1) -> None:
        """Estimate per-logit mean/std over WildChat samples at a chosen layer.

        The paper standardises each logit by its mean/std over 500 WildChat
        samples; we accumulate streaming statistics over the supplied texts.
        """
        import torch

        count = 0
        s1 = s2 = None
        for text in wildchat_texts:
            ids, hidden = self.client.hidden_states_for_text(text)  # [L+1,seq,d]
            resid = hidden[layer]                                   # [seq, d]
            logits = self.client.unembed(resid)                     # [seq, vocab]
            flat = logits.reshape(-1, logits.shape[-1])
            if s1 is None:
                s1 = flat.sum(0)
                s2 = (flat * flat).sum(0)
            else:
                s1 += flat.sum(0)
                s2 += (flat * flat).sum(0)
            count += flat.shape[0]
        self._mean = s1 / count
        var = s2 / count - self._mean ** 2
        self._std = var.clamp_min(1e-6).sqrt()

    # -- scoring -------------------------------------------------------------

    def _zscore_logits(self, logits):
        return (logits - self._mean) / self._std

    def score_text(self, text: str, *, layer_window=(30, 40)) -> EmotionScores:
        """Score every Ekman emotion per layer for a full text, averaged over
        positions, with the random-control mean regressed out."""
        import torch

        if self._mean is None:
            raise RuntimeError("Call calibrate(...) before scoring.")

        ids, hidden = self.client.hidden_states_for_text(text)   # [L+1, seq, d]
        n_layers = hidden.shape[0]
        per_layer: dict[str, list[float]] = {e: [] for e in EKMAN_EMOTIONS}

        for layer in range(n_layers):
            logits = self.client.unembed(hidden[layer])           # [seq, vocab]
            z = self._zscore_logits(logits)                       # [seq, vocab]
            # control signal: mean z over random tokens (per position), then
            # subtract its position-mean to regress out global drift.
            control = z[:, self.control_ids].mean(dim=1)          # [seq]
            control_mean = float(control.mean())
            for e in EKMAN_EMOTIONS:
                idx = self.emotion_ids[e]
                if not idx:
                    per_layer[e].append(0.0)
                    continue
                emo = z[:, idx].mean(dim=1)                        # [seq]
                signal = float(emo.mean()) - control_mean          # regress out
                per_layer[e].append(signal)

        lo, hi = layer_window
        hi = min(hi, n_layers - 1)
        agg = {e: float(sum(per_layer[e][lo:hi + 1]) / max(1, hi - lo + 1))
               for e in EKMAN_EMOTIONS}
        return EmotionScores(per_layer, agg)
