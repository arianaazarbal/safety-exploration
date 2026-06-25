"""Logit-based internal-emotion detection (Appendix I).

Method (faithful to Appendix I, with documented approximations):
  1. Classify the vocabulary into Ekman emotions (lexicon.py).
  2. For each layer, unembed the residual stream to obtain logits for emotion
     tokens (and a set of random tokens used to estimate common variance).
  3. Standardise each token's logit by its mean/std over WildChat baseline data.
  4. An emotion's score at a position/layer is the mean z-score over that
     emotion's tokens, after regressing out the common component estimated from
     random tokens (the paper notes all logits are correlated and rise/fall
     together over a conversation).

Requires an :class:`HFProvider` (hidden states + selective unembed). See
DESIGN.md for why we use a logit/lexicon approach rather than trained probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EKMAN_EMOTIONS, PROBE_AGG_LAYERS, PROBE_ZSCORE_SAMPLES
from ..models import Message
from ..models.local_hf import HFProvider
from .lexicon import classify_vocab


@dataclass
class ProbeBaseline:
    token_ids: list[int]  # all probed token ids (emotion + random), in order
    emotion_slices: dict[str, tuple[int, int]]  # emotion -> (start, end) into token_ids
    random_slice: tuple[int, int]
    mean: np.ndarray  # [n_layers, n_tokens]
    std: np.ndarray  # [n_layers, n_tokens]


class LogitEmotionProbe:
    def __init__(self, provider: HFProvider, *, n_random_tokens: int = 500, seed: int = 0):
        if not isinstance(provider, HFProvider):
            raise TypeError("LogitEmotionProbe requires an HFProvider")
        self.provider = provider
        self.tok = provider.tokenizer
        rng = np.random.default_rng(seed)

        vocab_by_emotion = classify_vocab(self.tok)
        token_ids: list[int] = []
        emotion_slices: dict[str, tuple[int, int]] = {}
        for emotion in EKMAN_EMOTIONS:
            ids = vocab_by_emotion[emotion]
            start = len(token_ids)
            token_ids.extend(ids)
            emotion_slices[emotion] = (start, len(token_ids))

        emotion_set = set(token_ids)
        candidates = [i for i in range(self.tok.vocab_size) if i not in emotion_set]
        random_ids = list(rng.choice(candidates, size=min(n_random_tokens, len(candidates)), replace=False))
        rand_start = len(token_ids)
        token_ids.extend(int(i) for i in random_ids)
        random_slice = (rand_start, len(token_ids))

        self.token_ids = token_ids
        self.emotion_slices = emotion_slices
        self.random_slice = random_slice
        self.baseline: ProbeBaseline | None = None

    # --- baseline standardisation ---------------------------------------- #
    def fit_baseline(self, wildchat_texts: list[str], *, max_positions: int = 64) -> ProbeBaseline:
        """Estimate per-token, per-layer logit mean/std over WildChat data."""
        per_layer_logits: list[list[np.ndarray]] = None  # type: ignore
        n_used = 0
        for text in wildchat_texts[:PROBE_ZSCORE_SAMPLES]:
            inputs = self.provider.token_ids([Message("user", text)])
            hs = self.provider.hidden_states(inputs["input_ids"])  # tuple [L+1] x [1,seq,d]
            if per_layer_logits is None:
                per_layer_logits = [[] for _ in range(len(hs))]
            seq = hs[0].shape[1]
            pos = np.linspace(0, seq - 1, min(max_positions, seq)).astype(int)
            for layer, h in enumerate(hs):
                logits = self.provider.selective_logits(h[0, pos], self.token_ids)
                per_layer_logits[layer].append(logits.float().cpu().numpy())
            n_used += 1

        n_layers = len(per_layer_logits)
        n_tokens = len(self.token_ids)
        mean = np.zeros((n_layers, n_tokens))
        std = np.ones((n_layers, n_tokens))
        for layer in range(n_layers):
            stacked = np.concatenate(per_layer_logits[layer], axis=0)  # [positions, tokens]
            mean[layer] = stacked.mean(axis=0)
            std[layer] = stacked.std(axis=0) + 1e-6
        self.baseline = ProbeBaseline(
            self.token_ids, self.emotion_slices, self.random_slice, mean, std
        )
        return self.baseline

    # --- scoring --------------------------------------------------------- #
    def score_messages(self, messages: list[Message], *, agg_layers=PROBE_AGG_LAYERS) -> dict:
        """Return per-emotion scores for a conversation.

        Output: {"per_layer": {emotion: np.ndarray[n_layers]},
                  "aggregated": {emotion: float}}  where aggregated averages
        z-scores over the agg_layers window and over all token positions.
        """
        if self.baseline is None:
            raise RuntimeError("Call fit_baseline() before scoring.")
        inputs = self.provider.token_ids(messages)
        hs = self.provider.hidden_states(inputs["input_ids"])
        n_layers = len(hs)
        rs, re_ = self.random_slice

        per_layer: dict[str, np.ndarray] = {e: np.zeros(n_layers) for e in EKMAN_EMOTIONS}
        for layer, h in enumerate(hs):
            logits = self.provider.selective_logits(h[0], self.token_ids).float().cpu().numpy()
            z = (logits - self.baseline.mean[layer]) / self.baseline.std[layer]  # [seq, tokens]
            common = z[:, rs:re_].mean(axis=1, keepdims=True)  # common component per position
            z_resid = z - common  # regress out shared variance (approximation)
            for emotion, (s, e) in self.emotion_slices.items():
                if e > s:
                    per_layer[emotion][layer] = float(z_resid[:, s:e].mean())

        lo, hi = agg_layers
        aggregated = {
            e: float(per_layer[e][lo:hi].mean()) if hi <= n_layers else float(per_layer[e].mean())
            for e in EKMAN_EMOTIONS
        }
        return {"per_layer": {e: per_layer[e].tolist() for e in EKMAN_EMOTIONS},
                "aggregated": aggregated}
