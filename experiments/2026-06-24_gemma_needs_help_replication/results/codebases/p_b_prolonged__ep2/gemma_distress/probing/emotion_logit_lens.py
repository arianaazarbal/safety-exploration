"""Logit-lens internal-emotion detection (Appendix I, Figures 14-15).

Method (Appendix I):
1. Classify vocabulary tokens into Ekman's six emotions (~1200 tokens total);
   see ``ekman_lexicon``.
2. Calibrate: over ~500 WildChat samples, unembed the residual stream at each
   layer and record the mean and std of each emotion-token logit (z-score
   reference).
3. Score a conversation: at each layer and token position, unembed the residual
   stream, z-score the emotion-token logits against the calibration stats,
   average within each emotion category, and regress out the component shared
   with a set of random control tokens (the paper notes all logits are
   correlated and rise/fall together over a conversation, so this removes the
   global drift, leaving an emotion-specific score).
4. Aggregate over layers 30-40 and a running window (paper uses 400-token
   windows over a ~12k-token conversation) for the conversation-level trace, or
   over token positions around the emotion onset for the layerwise view.

Comparing the vanilla instruct model with the DPO finetune on identical
responses shows whether DPO suppresses *internal* (not just expressed) emotion.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..probing.ekman_lexicon import EKMAN_EMOTIONS, build_emotion_token_ids
from ..models.hf_backend import HFBackend


@dataclass
class Calibration:
    token_ids: dict[str, list[int]]          # emotion -> token ids
    random_ids: list[int]                    # control tokens
    # Per-layer mean/std of logits for the union of (emotion + random) token ids.
    # Shape: (num_layers+1, n_tracked). `index_of` maps a token id to its column.
    mean: torch.Tensor
    std: torch.Tensor
    index_of: dict[int, int]


class EmotionLogitLens:
    def __init__(self, backend: HFBackend, n_random: int = 400,
                 max_per_emotion: int = 200, seed: int = 0):
        self.backend = backend
        self.tokenizer = backend.tokenizer
        rng = random.Random(seed)

        self.token_ids = build_emotion_token_ids(self.tokenizer, max_per_emotion)
        emo_union = sorted({i for ids in self.token_ids.values() for i in ids})
        vocab = self.tokenizer.vocab_size
        random_ids = []
        emo_set = set(emo_union)
        while len(random_ids) < n_random:
            t = rng.randrange(vocab)
            if t not in emo_set:
                random_ids.append(t)
        self.random_ids = random_ids

        self.tracked = sorted(set(emo_union) | set(random_ids))
        self.index_of = {tid: i for i, tid in enumerate(self.tracked)}
        self.calibration: Optional[Calibration] = None

    # ---------------------------------------------------------------
    @torch.no_grad()
    def _tracked_logits(self, text: str) -> torch.Tensor:
        """Return logits for tracked tokens at every layer/position.
        Shape: (num_layers+1, seq_len, n_tracked)."""
        resid = self.backend.residual_stream(text)        # (L+1, seq, hidden)
        logits = self.backend.unembed(resid)              # (L+1, seq, vocab)
        idx = torch.tensor(self.tracked, device=logits.device)
        return logits.index_select(-1, idx)

    @torch.no_grad()
    def calibrate(self, wildchat_texts: list[str]) -> Calibration:
        """Accumulate per-layer mean/std of tracked-token logits over WildChat."""
        sum_, sumsq, count = None, None, 0
        for text in wildchat_texts:
            tl = self._tracked_logits(text).float()       # (L+1, seq, T)
            s = tl.sum(dim=1)                              # (L+1, T)
            sq = (tl ** 2).sum(dim=1)
            sum_ = s if sum_ is None else sum_ + s
            sumsq = sq if sumsq is None else sumsq + sq
            count += tl.shape[1]
        mean = sum_ / max(count, 1)
        var = (sumsq / max(count, 1)) - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        self.calibration = Calibration(
            token_ids=self.token_ids, random_ids=self.random_ids,
            mean=mean.cpu(), std=std.cpu(), index_of=self.index_of)
        return self.calibration

    # ---------------------------------------------------------------
    @torch.no_grad()
    def score_text(self, text: str, regress_out_random: bool = True
                   ) -> dict[str, torch.Tensor]:
        """Per-layer, per-position emotion z-scores for `text`.

        Returns {emotion: tensor of shape (num_layers+1, seq_len)}.
        """
        if self.calibration is None:
            raise RuntimeError("Call calibrate(...) before score_text(...).")
        cal = self.calibration
        tl = self._tracked_logits(text).float().cpu()      # (L+1, seq, T)
        z = (tl - cal.mean.unsqueeze(1)) / cal.std.unsqueeze(1)

        # Global drift = mean z over the random control tokens.
        rand_cols = [cal.index_of[t] for t in cal.random_ids]
        drift = z[..., rand_cols].mean(dim=-1)             # (L+1, seq)

        out: dict[str, torch.Tensor] = {}
        for emotion, ids in self.token_ids.items():
            cols = [cal.index_of[t] for t in ids]
            score = z[..., cols].mean(dim=-1)              # (L+1, seq)
            if regress_out_random:
                score = score - drift
            out[emotion] = score
        return out

    # ---------------------------------------------------------------
    def conversation_trace(self, text: str, layers: tuple[int, int] = (30, 40),
                           window: int = 400) -> dict[str, list[float]]:
        """Running-average emotion trace over a conversation (Figure 14).

        Aggregates the per-position z-scores over `layers[0]:layers[1]`, then
        takes a running mean over `window` token positions.
        """
        scores = self.score_text(text)
        lo, hi = layers
        traces: dict[str, list[float]] = {}
        for emotion, mat in scores.items():
            layer_avg = mat[lo:hi].mean(dim=0)             # (seq,)
            traces[emotion] = _running_mean(layer_avg.tolist(), window)
        return traces

    def layerwise_at(self, text: str, position: int) -> dict[str, list[float]]:
        """Per-layer emotion z-scores at a single token position (Figure 15)."""
        scores = self.score_text(text)
        return {e: mat[:, position].tolist() for e, mat in scores.items()}


def _running_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1 or len(xs) <= 1:
        return xs
    out, acc = [], 0.0
    from collections import deque
    buf: deque = deque()
    for x in xs:
        buf.append(x)
        acc += x
        if len(buf) > window:
            acc -= buf.popleft()
        out.append(acc / len(buf))
    return out
