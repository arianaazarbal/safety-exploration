"""Logit-lens internal-emotion detection (Appendix I).

Method (Appendix I, second experiment): unembed the residual stream at each layer
and standardise each logit by its mean/std over 500 WildChat samples. For a given
emotion, average the z-scores over that emotion's tokens to get an emotion score
at each layer and each conversation position. Because all logits are correlated
and drift over a conversation, we additionally regress out the correlation with a
set of random tokens, leaving an emotion-specific residual.

We take this logit-based approach rather than training linear probes precisely to
avoid generating probe data (paper's stated rationale). The detector compares the
vanilla and DPO models on the same frustrated conversations, expecting the DPO
model to show suppressed negative-emotion scores at all depths.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..config import Config
from .emotion_tokens import EkmanLexicon, classify_vocab

log = logging.getLogger(__name__)


@dataclass
class EmotionTrajectory:
    """Per-layer, per-position emotion z-scores for one conversation."""

    emotion: str
    # scores[layer][position] = regressed emotion z-score
    scores: list[list[float]]

    def layer_means(self) -> list[float]:
        import numpy as np

        return [float(np.mean(layer)) if layer else float("nan") for layer in self.scores]


class InternalEmotionDetector:
    def __init__(self, hf_client, cfg: Config | None = None, lexicon: EkmanLexicon | None = None):
        """``hf_client`` must be a :class:`HuggingFaceClient` (needs activations)."""
        self.client = hf_client
        self.cfg = (cfg or Config.load("experiments")).get("internal_emotion", {})
        self.emotions = self.cfg.get(
            "ekman_emotions", ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
        )
        self.lexicon = lexicon or classify_vocab(hf_client.tokenizer, self.emotions)
        self._mean = None   # (num_layers, vocab) standardisation mean
        self._std = None
        self._random_token_ids: list[int] = []

    # ----------------------------------------------------------- calibration
    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, token) logit mean/std over WildChat samples.

        To bound memory we only track statistics for the union of emotion tokens
        plus a random-token control set, not the full vocabulary.
        """
        import numpy as np
        import torch

        n = int(self.cfg.get("standardisation_samples", 500))
        texts = wildchat_texts[:n]
        rng = random.Random(0)
        vocab = self.client.vocab_size()
        n_random = int(self.cfg.get("random_token_sample", 200))
        self._random_token_ids = rng.sample(range(vocab), n_random)
        tracked = sorted(set(self.lexicon.all_emotion_token_ids()) | set(self._random_token_ids))
        self._tracked = tracked
        tracked_idx = torch.tensor(tracked)

        sums = None
        sumsq = None
        count = 0
        for text in texts:
            layer_logits, _ = self.client.residual_logits(text)  # (L, S, V)
            sub = layer_logits.index_select(2, tracked_idx)       # (L, S, T)
            flat = sub.reshape(sub.shape[0], -1, sub.shape[2])    # (L, S, T)
            s = flat.sum(dim=1)          # (L, T)
            sq = (flat ** 2).sum(dim=1)  # (L, T)
            positions = flat.shape[1]
            sums = s if sums is None else sums + s
            sumsq = sq if sumsq is None else sumsq + sq
            count += positions
        mean = sums / max(count, 1)
        var = (sumsq / max(count, 1)) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        self._mean = mean.numpy()
        self._std = std.numpy()
        self._tracked_pos = {tid: i for i, tid in enumerate(tracked)}
        log.info("Calibrated logit stats over %d positions, %d tracked tokens.",
                 count, len(tracked))

    # --------------------------------------------------------------- scoring
    def trajectory(self, conversation_text: str, emotion: str) -> EmotionTrajectory:
        """Per-layer, per-position regressed z-score trajectory for one emotion."""
        import numpy as np

        if self._mean is None:
            raise RuntimeError("Call calibrate() before scoring.")
        layer_logits, _ = self._tracked_logits(conversation_text)  # (L, S, T)
        z = (layer_logits - self._mean[:, None, :]) / self._std[:, None, :]  # (L,S,T)

        emo_ids = self.lexicon.emotion_token_ids(emotion)
        emo_cols = [self._tracked_pos[t] for t in emo_ids if t in self._tracked_pos]
        rand_cols = [self._tracked_pos[t] for t in self._random_token_ids
                     if t in self._tracked_pos]
        if not emo_cols:
            return EmotionTrajectory(emotion, [[] for _ in range(z.shape[0])])

        emo_score = z[:, :, emo_cols].mean(axis=2)    # (L, S)
        baseline = z[:, :, rand_cols].mean(axis=2) if rand_cols else np.zeros_like(emo_score)

        # Regress out the random-token baseline per layer (residual emotion score).
        residual = np.empty_like(emo_score)
        for L in range(emo_score.shape[0]):
            y = emo_score[L]
            x = baseline[L]
            if self.cfg.get("regress_out_random_tokens", True) and np.std(x) > 1e-8:
                slope = np.cov(x, y)[0, 1] / np.var(x)
                intercept = y.mean() - slope * x.mean()
                residual[L] = y - (slope * x + intercept)
            else:
                residual[L] = y
        return EmotionTrajectory(emotion, [residual[L].tolist() for L in range(residual.shape[0])])

    def _tracked_logits(self, text: str):
        import torch

        layer_logits, ids = self.client.residual_logits(text)
        idx = torch.tensor(self._tracked)
        return layer_logits.index_select(2, idx).numpy(), ids

    def compare(
        self, conversation_text: str, emotions: list[str] | None = None
    ) -> dict[str, list[float]]:
        """Layer-mean z-score per emotion for one conversation."""
        emotions = emotions or self.emotions
        return {e: self.trajectory(conversation_text, e).layer_means() for e in emotions}
