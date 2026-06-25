"""Logit-based internal emotion detection (Appendix I).

Method (paper): classify the Gemma dictionary into Ekman's 6 emotions; for a
given residual-stream position, unembed (logit-lens) to get vocab logits;
standardise each logit by its mean/std over 500 WildChat samples; average the
z-scores over the tokens of an emotion category; regress out the shared
random-token drift to isolate per-emotion signal. Compare the vanilla instruct
model against the DPO finetune.

This module provides:
  * `EmotionProbe.fit_reference(...)` - estimate per-logit mean/std over WildChat,
  * `EmotionProbe.score(...)`        - per-layer per-emotion z-scores for a text,
  * a driver that contrasts vanilla vs DPO over a frustrated conversation.

It is the heaviest / most approximate component; see DESIGN.md for caveats.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .emotion_lexicon import EKMAN_EMOTIONS, build_emotion_token_ids

logger = logging.getLogger(__name__)


@dataclass
class ReferenceStats:
    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]
    random_token_ids: list[int]


class EmotionProbe:
    def __init__(self, client, layers: tuple[int, int] = (30, 40)):
        """client: a LocalGemmaClient (must support hidden_states)."""
        self.client = client
        self.layer_lo, self.layer_hi = layers
        self._emotion_ids = None
        self._reference: ReferenceStats | None = None

    # ---------------------------------------------------------------- internals
    def _logit_lens(self, hidden_states, model, tokenizer):
        """Apply final norm + unembedding to every layer's hidden states.

        Returns logits array [n_layers, seq, vocab] for the forward pass.
        """
        import torch

        norm = model.get_decoder().norm if hasattr(model, "get_decoder") else model.model.norm
        lm_head = model.get_output_embeddings()
        layer_logits = []
        with torch.no_grad():
            for h in hidden_states:  # tuple len n_layers+1
                normed = norm(h)
                logits = lm_head(normed)  # [batch, seq, vocab]
                layer_logits.append(logits[0].float().cpu().numpy())
        return np.stack(layer_logits, axis=0)  # [n_layers+1, seq, vocab]

    def _ensure_emotion_ids(self, tokenizer):
        if self._emotion_ids is None:
            self._emotion_ids = build_emotion_token_ids(tokenizer)
            counts = {e: len(v) for e, v in self._emotion_ids.items()}
            logger.info("emotion token counts: %s (total %d)", counts, sum(counts.values()))

    # ------------------------------------------------------------- reference
    def fit_reference(self, wildchat_texts: list[str], n_random: int = 500, seed: int = 0):
        """Estimate per-logit mean/std over WildChat (Appendix I)."""
        acc_sum = None
        acc_sqsum = None
        n_positions = 0
        tokenizer = model = None
        for text in wildchat_texts:
            messages = [{"role": "user", "content": text}]
            hs, tokenizer, model = self.client.hidden_states(messages)
            logits = self._logit_lens(hs, model, tokenizer)  # [L, seq, V]
            flat = logits.reshape(logits.shape[0], -1, logits.shape[2])  # [L, seq, V]
            s = flat.sum(axis=1)
            sq = (flat ** 2).sum(axis=1)
            acc_sum = s if acc_sum is None else acc_sum + s
            acc_sqsum = sq if acc_sqsum is None else acc_sqsum + sq
            n_positions += flat.shape[1]

        mean = acc_sum / max(1, n_positions)
        var = acc_sqsum / max(1, n_positions) - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))

        self._ensure_emotion_ids(tokenizer)
        rng = np.random.default_rng(seed)
        vocab = mean.shape[1]
        random_ids = rng.choice(vocab, size=min(n_random, vocab), replace=False).tolist()
        self._reference = ReferenceStats(mean=mean, std=std, random_token_ids=random_ids)
        return self._reference

    # ------------------------------------------------------------------ score
    def score(self, messages: list[dict], prefill: str | None = None) -> dict:
        """Return per-layer z-scores per emotion for the final-position residual,
        with shared random-token drift regressed out.
        """
        if self._reference is None:
            raise RuntimeError("call fit_reference() first")
        hs, tokenizer, model = self.client.hidden_states(messages, prefill=prefill)
        self._ensure_emotion_ids(tokenizer)
        logits = self._logit_lens(hs, model, tokenizer)  # [L, seq, V]
        # Use the final position (most recent token).
        z = (logits[:, -1, :] - self._reference.mean) / self._reference.std  # [L, V]

        # Shared drift = mean z over random tokens at each layer; subtract it.
        drift = z[:, self._reference.random_token_ids].mean(axis=1, keepdims=True)
        z_adj = z - drift

        out = {}
        for emotion in EKMAN_EMOTIONS:
            ids = self._emotion_ids[emotion]
            if not ids:
                out[emotion] = [float("nan")] * z_adj.shape[0]
                continue
            out[emotion] = z_adj[:, ids].mean(axis=1).tolist()  # per layer
        return out

    def band_score(self, messages: list[dict], prefill: str | None = None) -> dict:
        """Average each emotion's z-score over the configured layer band (30-40)."""
        per_layer = self.score(messages, prefill=prefill)
        out = {}
        for emotion, vals in per_layer.items():
            band = vals[self.layer_lo:self.layer_hi]
            band = [v for v in band if not np.isnan(v)]
            out[emotion] = float(np.mean(band)) if band else float("nan")
        return out
