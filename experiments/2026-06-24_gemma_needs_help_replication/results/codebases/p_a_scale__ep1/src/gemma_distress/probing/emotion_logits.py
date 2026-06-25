"""Logit-lens emotion detector (Appendix I).

To score an emotion at a token position and layer, we unembed the residual
stream, restrict to that emotion's tokens, standardise each token's logit by its
WildChat mean/std, average the z-scores, then subtract a random-token baseline
(a simple form of the paper's "regress out the correlation between random
tokens", which removes the global rise/fall of all logits over a conversation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from .lexicon import classify_vocabulary

log = get_logger("probing.emotion_logits")

EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]


class EmotionDetector:
    def __init__(self, provider, layers: list[int], per_emotion: int = 200):
        self.provider = provider
        self.layers = layers
        self.tokenizer = provider.tokenizer
        groups = classify_vocabulary(self.tokenizer, per_emotion)
        self.groups = groups
        # Flatten to a single vocab_subset; track column ranges per group.
        self.subset: list[int] = []
        self.ranges: dict[str, tuple[int, int]] = {}
        for name, ids in groups.items():
            start = len(self.subset)
            self.subset.extend(ids)
            self.ranges[name] = (start, len(self.subset))
        # Normalisation stats: per layer, per column.
        self.mean: dict[int, np.ndarray] = {}
        self.std: dict[int, np.ndarray] = {}

    # --- fit normalisation on WildChat ------------------------------------
    def fit(self, texts: list[str]) -> None:
        k = len(self.subset)
        sums = {l: np.zeros(k) for l in self.layers}
        sqs = {l: np.zeros(k) for l in self.layers}
        counts = {l: 0 for l in self.layers}
        for text in texts:
            res = self.provider.residual_logits(
                [{"role": "user", "content": text}],
                layers=self.layers, vocab_subset=self.subset,
            )
            for l in self.layers:
                arr = res["layers"][l]  # [seq, k]
                sums[l] += arr.sum(axis=0)
                sqs[l] += (arr ** 2).sum(axis=0)
                counts[l] += arr.shape[0]
        for l in self.layers:
            n = max(1, counts[l])
            mean = sums[l] / n
            var = np.maximum(sqs[l] / n - mean ** 2, 1e-6)
            self.mean[l] = mean
            self.std[l] = np.sqrt(var)
        log.info("Detector fitted on %d WildChat texts", len(texts))

    def save(self, path: str | Path) -> None:
        np.savez(path, subset=np.array(self.subset),
                 **{f"mean_{l}": self.mean[l] for l in self.layers},
                 **{f"std_{l}": self.std[l] for l in self.layers},
                 layers=np.array(self.layers))

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        for l in self.layers:
            self.mean[l] = data[f"mean_{l}"]
            self.std[l] = data[f"std_{l}"]

    # --- score a conversation ---------------------------------------------
    def score(self, messages: list[dict], prefill: str | None = None) -> dict:
        """Return per-layer arrays of emotion z-scores: {layer: {emotion: [seq]}},
        plus the token ids for positional windowing."""
        res = self.provider.residual_logits(
            messages, layers=self.layers, vocab_subset=self.subset, prefill=prefill,
        )
        out: dict[int, dict[str, np.ndarray]] = {}
        for l in self.layers:
            arr = res["layers"][l]  # [seq, k]
            z = (arr - self.mean[l]) / self.std[l]
            r0, r1 = self.ranges["_random"]
            baseline = z[:, r0:r1].mean(axis=1)  # [seq]
            emo_scores: dict[str, np.ndarray] = {}
            for emo in EMOTIONS:
                a, b = self.ranges[emo]
                if b > a:
                    emo_scores[emo] = z[:, a:b].mean(axis=1) - baseline
                else:
                    emo_scores[emo] = np.zeros(z.shape[0])
            out[l] = emo_scores
        return {"layers": out, "token_ids": res["token_ids"]}


def running_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = np.ones(min(window, len(values))) / min(window, len(values))
    return np.convolve(values, kernel, mode="same")
