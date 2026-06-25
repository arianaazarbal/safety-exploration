"""Logit-based internal emotion detection (Appendix I).

Method (Appendix I, verbatim procedure):
* Over the Gemma dictionary, words are classified into one or none of Ekman's 6
  basic emotions (~1200 emotion tokens) — see emotion_tokens.py.
* To score an emotion at a given residual-stream position/layer: unembed the
  residual stream (apply the LM head / final norm) to get logits, standardise
  each logit with its mean and std over 500 WildChat samples (z-score), then
  average the z-scores over that emotion's tokens.
* Logit values across tokens are correlated and drift over a conversation, so we
  additionally regress out the correlation with a set of random tokens to get a
  cleaner per-layer, per-position emotion score.
* Conversation-level scores aggregate over layers 30-40 with a running average
  over 400-token windows (Figure 14).

This module captures per-layer hidden states with a forward hook, applies the
unembedding, and computes standardised emotion scores. It does *not* train
probes (the paper deliberately uses this logit approach to avoid generating
probe data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ProbingConfig


@dataclass
class Standardisation:
    """Per-(layer, vocab) logit mean/std from WildChat, plus random-token basis."""

    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]
    random_token_ids: list[int]


class InternalEmotionProbe:
    def __init__(self, backend, emotion_token_ids: dict[str, list[int]], cfg: ProbingConfig):
        self.backend = backend          # HFBackend (exposes ._model, ._tokenizer)
        self.emotion_token_ids = emotion_token_ids
        self.cfg = cfg
        self._standardisation: Standardisation | None = None

    # ------------------------------------------------------------------ #
    # Residual-stream capture + unembedding
    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str) -> np.ndarray:
        """Return per-layer unembedded logits for every token position.

        Shape: [n_layers, seq_len, vocab]. Uses ``output_hidden_states`` to get
        the residual stream at each layer, applies the model's final norm + LM
        head (the unembedding) to each layer's hidden states.
        """
        import torch

        model = self.backend._model
        tok = self.backend._tokenizer
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple(n_layers+1) of [1, seq, d_model]

        lm_head = model.get_output_embeddings()
        norm = _final_norm(model)
        logits_per_layer = []
        for h in hidden[1:]:  # skip embedding layer
            normed = norm(h) if norm is not None else h
            logits = lm_head(normed)  # [1, seq, vocab]
            logits_per_layer.append(logits[0].float().cpu().numpy())
        return np.stack(logits_per_layer, axis=0)

    # ------------------------------------------------------------------ #
    # Standardisation over WildChat
    # ------------------------------------------------------------------ #
    def fit_standardisation(self, wildchat_texts: list[str], n_random_tokens: int = 200,
                            seed: int = 0) -> Standardisation:
        """Compute per-layer, per-vocab logit mean/std over WildChat samples."""
        samples = wildchat_texts[: self.cfg.n_wildchat_standardisation]
        running_sum = None
        running_sq = None
        count = 0
        for text in samples:
            ll = self._layer_logits(text)            # [L, seq, vocab]
            flat = ll.reshape(ll.shape[0], -1, ll.shape[-1])  # [L, seq, vocab]
            s = flat.sum(axis=1)                      # [L, vocab]
            sq = (flat ** 2).sum(axis=1)
            running_sum = s if running_sum is None else running_sum + s
            running_sq = sq if running_sq is None else running_sq + sq
            count += flat.shape[1]
        mean = running_sum / count
        var = np.maximum(running_sq / count - mean ** 2, 1e-8)
        std = np.sqrt(var)

        rng = np.random.default_rng(seed)
        vocab = mean.shape[-1]
        random_ids = rng.choice(vocab, size=min(n_random_tokens, vocab), replace=False).tolist()
        self._standardisation = Standardisation(mean=mean, std=std, random_token_ids=random_ids)
        return self._standardisation

    # ------------------------------------------------------------------ #
    # Emotion scoring
    # ------------------------------------------------------------------ #
    def emotion_scores(self, text: str) -> dict[str, np.ndarray]:
        """Per-emotion z-score trajectory aggregated over cfg.aggregate_layers.

        Returns {emotion: [score_per_token_position]} with the random-token
        correlation regressed out (if configured).
        """
        if self._standardisation is None:
            raise RuntimeError("Call fit_standardisation() first.")
        std_obj = self._standardisation
        ll = self._layer_logits(text)                # [L, seq, vocab]
        z = (ll - std_obj.mean[:, None, :]) / std_obj.std[:, None, :]  # [L, seq, vocab]

        lo, hi = self.cfg.aggregate_layers
        z_agg = z[lo:hi].mean(axis=0)                # [seq, vocab]

        # Baseline drift: mean z over random tokens at each position.
        if self.cfg.regress_out_random_tokens:
            baseline = z_agg[:, std_obj.random_token_ids].mean(axis=1, keepdims=True)
        else:
            baseline = 0.0

        scores = {}
        for emotion, ids in self.emotion_token_ids.items():
            if not ids:
                scores[emotion] = np.zeros(z_agg.shape[0])
                continue
            emo = z_agg[:, ids].mean(axis=1)         # [seq]
            scores[emotion] = emo - (baseline.squeeze(-1) if self.cfg.regress_out_random_tokens else 0.0)
        return scores

    def running_average(self, scores: np.ndarray) -> np.ndarray:
        """Running average over cfg.running_window_tokens (Figure 14)."""
        w = self.cfg.running_window_tokens
        if len(scores) <= w:
            return np.cumsum(scores) / (np.arange(len(scores)) + 1)
        kernel = np.ones(w) / w
        return np.convolve(scores, kernel, mode="valid")


def _final_norm(model):
    """Best-effort lookup of the model's final RMSNorm before the LM head."""
    for attr in ("model", "transformer"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "norm"):
            return inner.norm
    return getattr(model, "norm", None)
