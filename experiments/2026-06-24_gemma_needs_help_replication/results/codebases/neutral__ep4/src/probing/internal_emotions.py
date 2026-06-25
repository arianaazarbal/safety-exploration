"""Logit-based internal-emotion detection (Appendix I).

Method (following Appendix I):
  1. Classify Gemma vocabulary tokens into Ekman emotions via the seed lexicon.
  2. For a residual-stream activation at a layer/position, unembed it to logits
     over the vocabulary.
  3. Standardise each emotion-token logit to a z-score using its mean/std over
     ~500 WildChat samples (`fit_stats`).
  4. Average z-scores over the tokens in an emotion category.
  5. At the conversation level, all logits are correlated and drift together, so
     we regress out the correlation with a set of random reference tokens to
     isolate the emotion-specific signal (`_residualise`).

This avoids training linear probes (no probe data needed), matching the paper's
rationale. We note its limitations (Appendix I) in DESIGN.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (CACHE_DIR, EKMAN_EMOTIONS, PROBE_AGG_LAYERS,
                    PROBE_ZSCORE_SAMPLES)
from .emotion_lexicon import EKMAN_LEXICON


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocabulary token ids whose surface form
    matches a lexicon word (tolerant of the leading word-boundary marker)."""
    lex_to_emotion = {}
    for emo, words in EKMAN_LEXICON.items():
        for w in words:
            lex_to_emotion[w.lower()] = emo

    out = {e: [] for e in EKMAN_EMOTIONS}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        surface = tok.replace("▁", "").replace("Ġ", "").strip().lower()
        if surface in lex_to_emotion:
            out[lex_to_emotion[surface]].append(tid)
    return out


@dataclass
class ProbeStats:
    mean: np.ndarray          # [vocab] per-token logit mean
    std: np.ndarray           # [vocab] per-token logit std
    random_token_ids: list[int]


def fit_stats(model, wildchat_texts: list[str], *, layer: int,
              n_samples: int = PROBE_ZSCORE_SAMPLES,
              n_random_tokens: int = 200, seed: int = 0) -> ProbeStats:
    """Estimate per-token logit mean/std at a given layer over WildChat text."""
    import torch

    rng = random.Random(seed)
    sums = None
    sqs = None
    count = 0
    for text in wildchat_texts[:n_samples]:
        hidden, _ = model.residual_stream(text)          # [L+1, T, D]
        logits = model.unembed(hidden[layer]).float()    # [T, V]
        l = logits.detach().cpu().numpy()
        if sums is None:
            V = l.shape[1]
            sums = np.zeros(V); sqs = np.zeros(V)
        sums += l.sum(axis=0)
        sqs += (l ** 2).sum(axis=0)
        count += l.shape[0]
    mean = sums / max(1, count)
    var = np.maximum(sqs / max(1, count) - mean ** 2, 1e-6)
    std = np.sqrt(var)
    V = mean.shape[0]
    random_ids = rng.sample(range(V), min(n_random_tokens, V))
    return ProbeStats(mean=mean, std=std, random_token_ids=random_ids)


def _zscores(logits_row: np.ndarray, stats: ProbeStats) -> np.ndarray:
    return (logits_row - stats.mean) / stats.std


def _residualise(emotion_z: np.ndarray, random_z_mean: np.ndarray) -> np.ndarray:
    """Regress out the shared drift captured by the mean random-token z-score.

    Fits emotion_z ~ a + b * random_z_mean across positions and returns the
    residual (the emotion-specific component)."""
    if len(emotion_z) < 3:
        return emotion_z - random_z_mean
    A = np.vstack([np.ones_like(random_z_mean), random_z_mean]).T
    coef, *_ = np.linalg.lstsq(A, emotion_z, rcond=None)
    return emotion_z - A @ coef


def emotion_trajectory(model, text: str, emotion_token_ids: dict[str, list[int]],
                       stats: ProbeStats, *, layer: int) -> dict[str, np.ndarray]:
    """Return per-position residualised z-score trajectory for each emotion."""
    hidden, _ = model.residual_stream(text)
    logits = model.unembed(hidden[layer]).float().detach().cpu().numpy()  # [T, V]

    # mean random-token z over positions (the shared drift signal)
    rand_z = np.stack([_zscores(logits[t], stats)[stats.random_token_ids].mean()
                       for t in range(logits.shape[0])])

    out = {}
    for emo, ids in emotion_token_ids.items():
        if not ids:
            out[emo] = np.zeros(logits.shape[0]); continue
        per_pos = np.stack([_zscores(logits[t], stats)[ids].mean()
                            for t in range(logits.shape[0])])
        out[emo] = _residualise(per_pos, rand_z)
    return out


def aggregate_layers_trajectory(model, text: str, emotion_token_ids, stats_by_layer,
                                layers=PROBE_AGG_LAYERS) -> dict[str, np.ndarray]:
    """Average emotion trajectories over a layer window (Fig 14 uses 30-40)."""
    lo, hi = layers
    acc = None
    for layer in range(lo, hi):
        traj = emotion_trajectory(model, text, emotion_token_ids,
                                  stats_by_layer[layer], layer=layer)
        if acc is None:
            acc = {e: [] for e in traj}
        for e, v in traj.items():
            acc[e].append(v)
    return {e: np.mean(np.stack(vs), axis=0) for e, vs in acc.items()}


if __name__ == "__main__":
    # Minimal smoke-test entry point (requires a loaded model + WildChat slice).
    print("internal_emotions: import-only module; invoke via scripts/08_run_probing.py")
