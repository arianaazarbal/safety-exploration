"""Logit-lens internal emotion detection (Appendix I).

Method (Appendix I):
  * Unembed the residual stream at each layer to get per-token vocab logits.
  * Standardise each vocab logit by its mean/std over ~500 WildChat samples
    (per layer), giving z-scores.
  * For an emotion, average the z-scores over that emotion's tokens.
  * Because all logits co-move over a conversation, regress out the common mode
    estimated from a random token set (we subtract the mean random-token z-score
    at each layer/position — documented as our concrete form of "regress out").

Produces, for a target text, an (emotion x layer) or (emotion x position) signal
that can be aggregated over layers 30-40 (config) and windowed over tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class StandardisationStats:
    """Per-layer, per-vocab mean/std of logits over the reference corpus."""
    mean: torch.Tensor   # (layers, vocab)
    std: torch.Tensor    # (layers, vocab)

    def save(self, path: str | Path) -> None:
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardisationStats":
        d = torch.load(path, map_location="cpu")
        return cls(mean=d["mean"], std=d["std"])


def compute_standardisation(hf_model, texts: list[str]) -> StandardisationStats:
    """Running mean/std of per-layer logits over all token positions in ``texts``."""
    count = 0
    mean = None
    m2 = None  # sum of squared deviations (Welford)
    for text in texts:
        logits = hf_model.residual_logit_lens(text).float().cpu()  # (L, S, V)
        L, S, V = logits.shape
        flat = logits.permute(1, 0, 2).reshape(S, L, V)  # (S, L, V): each token = sample
        for s in range(S):
            x = flat[s]  # (L, V)
            count += 1
            if mean is None:
                mean = torch.zeros_like(x)
                m2 = torch.zeros_like(x)
            delta = x - mean
            mean += delta / count
            m2 += delta * (x - mean)
    std = torch.sqrt(m2 / max(1, count - 1)).clamp_min(1e-6)
    return StandardisationStats(mean=mean, std=std)


def _zscores(hf_model, text: str, stats: StandardisationStats) -> torch.Tensor:
    logits = hf_model.residual_logit_lens(text).float().cpu()  # (L, S, V)
    z = (logits - stats.mean.unsqueeze(1)) / stats.std.unsqueeze(1)
    return z  # (L, S, V)


def emotion_scores(
    hf_model,
    text: str,
    stats: StandardisationStats,
    emotion_tokens: dict[str, list[int]],
    random_tokens: list[int],
    layer_range: tuple[int, int] = (30, 40),
) -> dict[str, np.ndarray]:
    """Return per-emotion score per token position, aggregated over layer_range.

    score = mean_z(emotion tokens) - mean_z(random tokens)  [common-mode removal]
    """
    z = _zscores(hf_model, text, stats)  # (L, S, V)
    lo, hi = layer_range
    z = z[lo:hi]  # (l, S, V)
    rand = z[:, :, random_tokens].mean(dim=2)  # (l, S) common mode
    out = {}
    for emotion, ids in emotion_tokens.items():
        if not ids:
            out[emotion] = np.zeros(z.shape[1])
            continue
        emo = z[:, :, ids].mean(dim=2)         # (l, S)
        score = (emo - rand).mean(dim=0)       # (S,) averaged over layers
        out[emotion] = score.numpy()
    return out


def windowed_running_average(scores: np.ndarray, window: int = 400) -> np.ndarray:
    """Running average over a token window (Figure 14 plots windows of 400)."""
    if len(scores) == 0:
        return scores
    kernel = np.ones(min(window, len(scores))) / min(window, len(scores))
    return np.convolve(scores, kernel, mode="same")


def compare_models(
    vanilla_hf, dpo_hf, text: str,
    stats_vanilla: StandardisationStats, stats_dpo: StandardisationStats,
    emotion_tokens: dict[str, list[int]], random_tokens: list[int],
    layer_range=(30, 40), out_path: Optional[str | Path] = None,
) -> dict:
    """Compute emotion trajectories for the same text in both models (Figure 14)."""
    result = {
        "vanilla": {e: windowed_running_average(v).tolist()
                    for e, v in emotion_scores(vanilla_hf, text, stats_vanilla,
                                                emotion_tokens, random_tokens,
                                                layer_range).items()},
        "dpo": {e: windowed_running_average(v).tolist()
                for e, v in emotion_scores(dpo_hf, text, stats_dpo,
                                           emotion_tokens, random_tokens,
                                           layer_range).items()},
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(out_path, "w"))
    return result
