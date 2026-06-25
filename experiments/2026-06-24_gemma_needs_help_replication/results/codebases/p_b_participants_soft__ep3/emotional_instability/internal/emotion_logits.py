"""Logit-based internal emotion detection (Appendix I, Figures 14-15).

Method (paper): for a given emotion, unembed the residual stream at each layer,
standardise each emotion-token logit by its mean/std over 500 WildChat samples,
average the z-scores over the emotion's token set. Because all logits are
correlated and drift over a conversation, we additionally regress out the
correlation with random tokens, giving an emotion score at each layer and each
conversation position.

This is a "logit lens" probe: it applies the model's final norm + unembedding
(lm_head) to *intermediate* layer activations. No probe training required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..config import INTERNAL_PROBE
from .ekman_tokens import build_emotion_token_ids


def _find_final_norm(model):
    """Locate the model's final RMSNorm (architecture-tolerant)."""
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "norm"):
            return base.norm
    if hasattr(model, "norm"):
        return model.norm
    return None  # fall back to identity


@torch.no_grad()
def _layerwise_logits(model, input_ids) -> torch.Tensor:
    """Return logits [n_layers, seq, vocab] from the logit lens at every layer."""
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states  # tuple length n_layers+1 (embeddings + each block)
    norm = _find_final_norm(model)
    lm_head = model.get_output_embeddings()
    logits_per_layer = []
    for h in hidden[1:]:  # skip embedding layer
        hn = norm(h) if norm is not None else h
        logits_per_layer.append(lm_head(hn)[0])  # [seq, vocab]
    return torch.stack(logits_per_layer, dim=0)  # [n_layers, seq, vocab]


@dataclass
class StandardisationStats:
    # Per-layer, per-vocab mean/std of logits, estimated over WildChat data.
    mean: torch.Tensor  # [n_layers, vocab]
    std: torch.Tensor   # [n_layers, vocab]


def compute_standardisation_stats(
    model,
    tokenizer,
    wildchat_texts: list[str],
    n_samples: int = INTERNAL_PROBE["standardisation_samples"],
    max_len: int = 512,
    device: Optional[str] = None,
) -> StandardisationStats:
    """Estimate per-layer per-vocab logit mean/std over `n_samples` WildChat
    texts (running mean/var so memory stays bounded)."""
    device = device or next(model.parameters()).device
    sum_, sumsq, count = None, None, 0
    for text in wildchat_texts[:n_samples]:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        logits = _layerwise_logits(model, ids).float()      # [L, seq, V]
        s = logits.sum(dim=1)                                # [L, V]
        ss = (logits ** 2).sum(dim=1)                        # [L, V]
        sum_ = s if sum_ is None else sum_ + s
        sumsq = ss if sumsq is None else sumsq + ss
        count += logits.shape[1]
    mean = sum_ / count
    var = (sumsq / count) - mean ** 2
    std = torch.clamp(var, min=1e-6).sqrt()
    return StandardisationStats(mean=mean.cpu(), std=std.cpu())


class EmotionLogitDetector:
    def __init__(self, model, tokenizer, stats: StandardisationStats,
                 emotion_token_ids: Optional[dict[str, list[int]]] = None,
                 n_random_tokens: int = 500, seed: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats
        self.emotion_token_ids = emotion_token_ids or build_emotion_token_ids(tokenizer)
        rng = np.random.default_rng(seed)
        vocab = stats.mean.shape[1]
        self.random_ids = rng.choice(vocab, size=min(n_random_tokens, vocab), replace=False)

    @torch.no_grad()
    def score_text(self, text: str, max_len: int = 12000) -> dict[str, np.ndarray]:
        """Return {emotion: [n_layers, seq]} z-scores with random-token drift
        regressed out, for one (possibly long) conversation string."""
        device = next(self.model.parameters()).device
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_len).input_ids.to(device)
        logits = _layerwise_logits(self.model, ids).float().cpu()  # [L, seq, V]
        z = (logits - self.stats.mean[:, None, :]) / self.stats.std[:, None, :]

        # Baseline drift = mean z over random tokens, per (layer, position).
        baseline = z[:, :, self.random_ids].mean(dim=2)            # [L, seq]

        out = {}
        for emotion, tok_ids in self.emotion_token_ids.items():
            if not tok_ids:
                out[emotion] = np.zeros((z.shape[0], z.shape[1]))
                continue
            emo_z = z[:, :, tok_ids].mean(dim=2)                   # [L, seq]
            # Regress out the baseline correlation per layer (residualise).
            out[emotion] = _residualise(emo_z.numpy(), baseline.numpy())
        return out

    def conversation_trajectory(
        self,
        text: str,
        layers: tuple[int, int] = INTERNAL_PROBE["aggregate_layers"],
        window: int = INTERNAL_PROBE["running_window_tokens"],
    ) -> dict[str, np.ndarray]:
        """Conversation-level emotion trajectory: aggregate over layers
        [lo, hi) and take a running average over `window` tokens (Figure 14)."""
        scores = self.score_text(text)
        lo, hi = layers
        traj = {}
        for emotion, arr in scores.items():
            layer_avg = arr[lo:hi].mean(axis=0)                    # [seq]
            traj[emotion] = _running_mean(layer_avg, window)
        return traj


def _residualise(emo: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Per-layer OLS residual of emotion z-scores on the random-token baseline."""
    out = np.empty_like(emo)
    for L in range(emo.shape[0]):
        x = baseline[L]
        y = emo[L]
        if np.std(x) < 1e-8:
            out[L] = y - y.mean()
            continue
        beta = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        alpha = y.mean() - beta * x.mean()
        out[L] = y - (alpha + beta * x)
    return out


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x, kernel, mode="same")
