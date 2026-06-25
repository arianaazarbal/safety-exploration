"""Logit-based internal emotion detection (Appendix I).

Method (from the paper):
  * Classify the vocabulary into Ekman's 6 emotions (lexicon.py).
  * For a residual-stream vector at a given layer, unembed it (final norm +
    lm_head) to logits, then standardise each logit by its mean/std over 500
    WildChat samples (calibration). The emotion score at that layer/token is the
    mean standardised logit (z-score) over the emotion's tokens.
  * Because all logits are correlated and drift over a conversation, we regress
    out a "random token" component (mean z over a control token pool) to isolate
    emotion-specific signal.

This operates on a HFModel (needs residual_stream + unembed). It is Gemma-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .lexicon import EKMAN_EMOTIONS, build_emotion_token_ids


@dataclass
class Calibration:
    """Per-layer mean/std of each tracked logit, over calibration tokens."""

    layers: list[int]
    token_ids: list[int]                       # union of emotion + control ids
    mean: dict                                  # layer -> tensor [n_tokens]
    std: dict                                   # layer -> tensor [n_tokens]
    emotion_to_ids: dict
    control_ids: list[int]
    id_to_pos: dict = field(default_factory=dict)


class InternalEmotionDetector:
    def __init__(self, hf_model, layers: Optional[list[int]] = None):
        self.model = hf_model
        self.torch = hf_model.torch
        n_layers = self.model.model.config.num_hidden_layers
        # hidden_states index: 0=embeddings, i=output of layer i-1. We track the
        # decoder-layer outputs 1..n_layers (so "layer L" == hidden_states[L+1]).
        self.layers = layers or list(range(n_layers))
        self.emotion_to_ids, self.control_ids = build_emotion_token_ids(self.model.tokenizer)
        # Tracked vocab subset for efficient z-scoring.
        ids = sorted(
            {tid for ids in self.emotion_to_ids.values() for tid in ids} | set(self.control_ids)
        )
        self.token_ids = ids
        self.id_to_pos = {tid: i for i, tid in enumerate(ids)}
        self.calibration: Optional[Calibration] = None

    # ---- calibration ------------------------------------------------------
    def calibrate(self, texts: list[str]) -> Calibration:
        """Estimate per-layer mean/std of tracked logits over calibration texts
        (paper: 500 WildChat samples)."""
        torch = self.torch
        sums = {L: None for L in self.layers}
        sqs = {L: None for L in self.layers}
        counts = {L: 0 for L in self.layers}
        idx = torch.tensor(self.token_ids, device=self.model.model.device)

        for text in texts:
            _, hs = self.model.residual_stream(text)
            for L in self.layers:
                hidden = hs[L + 1]                       # [seq, hidden]
                logits = self.model.unembed(hidden)[:, idx]  # [seq, n_tracked]
                s = logits.sum(dim=0)
                sq = (logits ** 2).sum(dim=0)
                sums[L] = s if sums[L] is None else sums[L] + s
                sqs[L] = sq if sqs[L] is None else sqs[L] + sq
                counts[L] += logits.shape[0]

        mean, std = {}, {}
        for L in self.layers:
            n = max(1, counts[L])
            m = sums[L] / n
            var = (sqs[L] / n) - m ** 2
            mean[L] = m
            std[L] = var.clamp_min(1e-8).sqrt()
        self.calibration = Calibration(
            layers=self.layers,
            token_ids=self.token_ids,
            mean=mean,
            std=std,
            emotion_to_ids=self.emotion_to_ids,
            control_ids=self.control_ids,
            id_to_pos=self.id_to_pos,
        )
        return self.calibration

    # ---- scoring ----------------------------------------------------------
    def score_text(self, text: str, layers: Optional[list[int]] = None) -> dict:
        """Return {emotion: {layer: z_score}} averaged over all tokens in `text`,
        with the random-token component regressed out (subtracted)."""
        assert self.calibration is not None, "call calibrate() first"
        torch = self.torch
        layers = layers or self.layers
        idx = torch.tensor(self.token_ids, device=self.model.model.device)

        _, hs = self.model.residual_stream(text)
        emo_pos = {
            e: torch.tensor([self.id_to_pos[t] for t in ids], device=idx.device)
            for e, ids in self.emotion_to_ids.items()
            if ids
        }
        ctrl_pos = torch.tensor(
            [self.id_to_pos[t] for t in self.control_ids], device=idx.device
        )

        out: dict[str, dict[int, float]] = {e: {} for e in EKMAN_EMOTIONS}
        for L in layers:
            hidden = hs[L + 1]
            logits = self.model.unembed(hidden)[:, idx]            # [seq, n_tracked]
            z = (logits - self.calibration.mean[L]) / self.calibration.std[L]
            z_mean_tokens = z.mean(dim=0)                          # [n_tracked]
            control = z_mean_tokens[ctrl_pos].mean()              # scalar drift
            for e, pos in emo_pos.items():
                out[e][L] = float(z_mean_tokens[pos].mean() - control)
        return out

    def score_trajectory(self, text: str, window_tokens: int = 400,
                         layers: Optional[list[int]] = None) -> list[dict]:
        """Sliding-window emotion scores over a long conversation (Figure 14).

        Returns a list of {token_pos, scores: {emotion: {layer: z}}} computed on
        successive `window_tokens`-sized chunks of the running residual stream.
        """
        assert self.calibration is not None
        torch = self.torch
        layers = layers or self.layers
        idx = torch.tensor(self.token_ids, device=self.model.model.device)
        ids, hs = self.model.residual_stream(text)
        seq = ids.shape[0]

        ctrl_pos = torch.tensor([self.id_to_pos[t] for t in self.control_ids], device=idx.device)
        emo_pos = {
            e: torch.tensor([self.id_to_pos[t] for t in ed], device=idx.device)
            for e, ed in self.emotion_to_ids.items() if ed
        }

        results = []
        for start in range(0, seq, window_tokens):
            end = min(seq, start + window_tokens)
            frame = {e: {} for e in EKMAN_EMOTIONS}
            for L in layers:
                hidden = hs[L + 1][start:end]
                logits = self.model.unembed(hidden)[:, idx]
                z = ((logits - self.calibration.mean[L]) / self.calibration.std[L]).mean(dim=0)
                control = z[ctrl_pos].mean()
                for e, pos in emo_pos.items():
                    frame[e][L] = float(z[pos].mean() - control)
            results.append({"token_pos": end, "scores": frame})
        return results

    def save_calibration(self, path: Path) -> None:
        import torch

        torch.save(
            {
                "layers": self.layers,
                "token_ids": self.token_ids,
                "mean": {L: self.calibration.mean[L].cpu() for L in self.layers},
                "std": {L: self.calibration.std[L].cpu() for L in self.layers},
                "emotion_to_ids": self.emotion_to_ids,
                "control_ids": self.control_ids,
            },
            path,
        )
