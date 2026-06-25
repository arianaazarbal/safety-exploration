"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing the paper):
  * Over the Gemma dictionary, classify words into Ekman's 6 emotions (ekman.py).
  * To score a given emotion at a layer/position: unembed the residual stream
    (apply final norm + lm_head to that layer's hidden state), z-score each
    emotion-token logit using its mean/std over 500 WildChat samples, then
    average the z-scores over all tokens in the emotion category.
  * For conversation-level detection, the logits are all correlated and drift
    over a conversation, so we additionally regress out the correlation with a
    set of random tokens to isolate the emotion signal.
  * Conversation trajectory: aggregate over layers 30-40, running average over
    400-token windows.
  * Layerwise stages: average over tokens at three points — 40-20 tokens before
    onset, 0-20 tokens before onset, and the final 20 tokens.

This is a faithful re-implementation of the described procedure; the paper does
not release exact code, so a few details (random-token count, regression form)
are chosen and documented in DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from ..models.hf_local import HFLocalClient
from .ekman import build_emotion_token_dictionary


@dataclass
class Calibration:
    """Per-token logit mean/std at each layer, estimated over WildChat samples,
    plus the random-token id set used for correlation removal."""
    layers: list[int]
    mean: dict[int, np.ndarray]      # layer -> [vocab] mean logit
    std: dict[int, np.ndarray]       # layer -> [vocab] std logit
    random_token_ids: list[int]


class EmotionDetector:
    def __init__(self, client: HFLocalClient, layers: Optional[list[int]] = None):
        self.client = client
        self.layers = layers or list(range(0, client.model.config.num_hidden_layers + 1))
        self.emotion_tokens = build_emotion_token_dictionary(client.tokenizer)
        self.calibration: Optional[Calibration] = None

    # ------------------------------------------------------------------ #
    # Calibration over WildChat
    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str], *,
                  n_random_tokens: int = 200, seed: int = 0) -> Calibration:
        """Estimate per-token logit mean/std at each layer over WildChat samples.

        We accumulate sum and sum-of-squares of the unembedded logits over all
        token positions of all calibration texts (per layer), then derive
        mean/std. `n_random_tokens` random vocab ids are stored for the
        correlation-removal step in `score_text`.
        """
        rng = np.random.default_rng(seed)
        n = min(len(wildchat_texts), config.INTERNAL.zscore_calibration_samples)
        return self._calibrate_clean(wildchat_texts[:n], n_random_tokens, rng)

    def _calibrate_clean(self, texts, n_random_tokens, rng) -> Calibration:
        layers = self.layers
        vocab = self.client.model.config.vocab_size
        # Collect all per-position logit vectors per layer (memory-bounded by
        # subsampling positions if needed).
        sums = {l: np.zeros(vocab, dtype=np.float64) for l in layers}
        sqs = {l: np.zeros(vocab, dtype=np.float64) for l in layers}
        counts = {l: 0 for l in layers}
        for text in texts:
            logits_by_layer, _ = self.client.residual_stream_logits(text, layers)
            for l in layers:
                lg = logits_by_layer[l].numpy().astype(np.float64)
                sums[l] += lg.sum(axis=0)
                sqs[l] += (lg ** 2).sum(axis=0)
                counts[l] += lg.shape[0]
        mean, std = {}, {}
        for l in layers:
            c = max(1, counts[l])
            mu = sums[l] / c
            var = np.maximum(sqs[l] / c - mu ** 2, 1e-8)
            mean[l] = mu
            std[l] = np.sqrt(var)
        random_token_ids = list(rng.integers(0, vocab, size=n_random_tokens))
        self.calibration = Calibration(layers, mean, std, random_token_ids)
        return self.calibration

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _zscored(self, logits: np.ndarray, layer: int) -> np.ndarray:
        cal = self.calibration
        return (logits - cal.mean[layer]) / cal.std[layer]

    def score_text(self, text: str, *, regress_random: bool = True) -> dict:
        """Return per-layer, per-position emotion z-score arrays plus token ids.

        Output: {emotion: {layer: np.ndarray[seq]}}, token_ids.
        """
        assert self.calibration is not None, "call calibrate() first"
        layers = self.layers
        logits_by_layer, token_ids = self.client.residual_stream_logits(text, layers)
        token_ids = token_ids.numpy()

        out: dict[str, dict[int, np.ndarray]] = {e: {} for e in self.emotion_tokens}
        for l in layers:
            z = self._zscored(logits_by_layer[l].numpy().astype(np.float64), l)  # [seq, vocab]
            # Random-token baseline (mean z over random tokens) for de-correlation.
            if regress_random:
                base = z[:, self.calibration.random_token_ids].mean(axis=1)  # [seq]
            else:
                base = np.zeros(z.shape[0])
            for emotion, tok_ids in self.emotion_tokens.items():
                if not tok_ids:
                    out[emotion][l] = np.zeros(z.shape[0])
                    continue
                emo = z[:, tok_ids].mean(axis=1)        # avg z over emotion tokens
                out[emotion][l] = emo - base            # regress out common drift
        return {"scores": out, "token_ids": token_ids}


def conversation_trajectory(detector: EmotionDetector, text: str, *,
                            layers: tuple[int, int] = config.INTERNAL.aggregate_layers,
                            window: int = config.INTERNAL.running_window_tokens) -> dict:
    """Figure 14: emotion z-score trajectory aggregated over layers [lo, hi),
    running-averaged over `window`-token windows."""
    res = detector.score_text(text)
    lo, hi = layers
    used = [l for l in detector.layers if lo <= l < hi]
    traj = {}
    for emotion, by_layer in res["scores"].items():
        stacked = np.stack([by_layer[l] for l in used], axis=0).mean(axis=0)  # [seq]
        # running average
        if window > 1 and len(stacked) >= 1:
            kernel = np.ones(min(window, len(stacked))) / min(window, len(stacked))
            stacked = np.convolve(stacked, kernel, mode="same")
        traj[emotion] = stacked
    return traj


def layerwise_stages(detector: EmotionDetector, text: str, onset_token_index: int) -> dict:
    """Figure 15: per-layer emotion z-scores averaged over tokens at three stages
    relative to emotion onset: [-40,-20), [-20,0) before onset, and the final 20
    tokens."""
    res = detector.score_text(text)
    n = len(res["token_ids"])
    stages = {
        "pre_40_20": (max(0, onset_token_index - 40), max(0, onset_token_index - 20)),
        "pre_20_0": (max(0, onset_token_index - 20), max(0, onset_token_index)),
        "final_20": (max(0, n - 20), n),
    }
    out: dict[str, dict[str, dict[int, float]]] = {}
    for emotion, by_layer in res["scores"].items():
        out[emotion] = {}
        for stage, (a, b) in stages.items():
            out[emotion][stage] = {
                l: float(by_layer[l][a:b].mean()) if b > a else float("nan")
                for l in detector.layers
            }
    return out
