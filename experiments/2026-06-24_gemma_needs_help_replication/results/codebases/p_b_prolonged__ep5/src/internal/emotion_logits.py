"""Logit-based internal emotion detection (Appendix I).

Method (per the paper):
  1. Classify ~1200 vocab tokens into Ekman's 6 emotions (emotion_tokens.py).
  2. For a given text, unembed the residual stream at each layer to get per-layer,
     per-token logits over the vocabulary.
  3. Standardise each emotion-token logit using its mean/std computed over 500
     WildChat samples (z-score), then average z-scores over the tokens in an
     emotion category to get that emotion's score at each layer / position.
  4. For conversation-level detection, regress out the shared (all-logits-rise-
     and-fall) component using random control tokens, leaving a residual emotion
     score per layer per conversation position.

We avoid linear probes (no probe-training data needed), matching the paper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR
from .emotion_tokens import EKMAN, build_emotion_token_ids

N_CALIBRATION_SAMPLES = 500           # WildChat samples for z-score stats
N_CONTROL_TOKENS = 200                # random tokens for the regress-out baseline


@dataclass
class EmotionStats:
    """Per-token mean/std of logits over the calibration corpus, plus the random
    control-token id set used to regress out the shared component."""
    mean: dict                         # {token_id: float}  (stored as np arrays at runtime)
    std: dict
    control_ids: list


def calibrate(model, tokenizer, *, layers: list[int],
              calib_texts: list[str], out_path: Optional[Path] = None):
    """Compute per-token logit mean/std over `calib_texts` at the given layers and
    a random control-token set. Persists to disk for reuse across models so the
    same standardisation is applied to vanilla and DPO models."""
    import numpy as np
    out_path = out_path or (ARTIFACTS_DIR / "emotion_calibration.npz")

    emo_ids = build_emotion_token_ids(tokenizer)
    all_emo_ids = sorted({i for ids in emo_ids.values() for i in ids})

    # random control tokens (deterministic choice for reproducibility)
    rng = np.random.default_rng(0)
    control_ids = sorted(rng.choice(tokenizer.vocab_size, size=N_CONTROL_TOKENS,
                                    replace=False).tolist())
    track_ids = sorted(set(all_emo_ids) | set(control_ids))

    # Online mean/var over (layer-averaged) logits for tracked tokens.
    sums = np.zeros(len(track_ids))
    sqs = np.zeros(len(track_ids))
    count = 0
    id_to_col = {tid: c for c, tid in enumerate(track_ids)}

    for text in calib_texts[:N_CALIBRATION_SAMPLES]:
        layer_logits, _ = model.unembed_residual(text)         # [L+1, T, V]
        sel = layer_logits[layers].float().mean(0)             # [T, V] avg over layers
        sub = sel[:, track_ids].cpu().numpy()                  # [T, len(track)]
        sums += sub.sum(0)
        sqs += (sub ** 2).sum(0)
        count += sub.shape[0]

    mean = sums / max(count, 1)
    var = np.maximum(sqs / max(count, 1) - mean ** 2, 1e-8)
    std = np.sqrt(var)

    # Save each emotion's token-id list as its own array (np.savez can't pickle a
    # dict); reload by the same "emo_<name>" key convention.
    emo_arrays = {f"emo_{e}": np.array(emo_ids[e], dtype=np.int64) for e in EKMAN}
    np.savez(out_path, track_ids=np.array(track_ids), mean=mean, std=std,
             control_cols=np.array([id_to_col[c] for c in control_ids]),
             **emo_arrays)
    return out_path


def score_text_emotions(model, tokenizer, text: str, *, layers: list[int],
                        calibration_path: Path) -> dict:
    """Return {emotion: mean_z_score} for `text`, aggregated over `layers` and all
    tokens, with the shared component regressed out via control tokens."""
    import numpy as np
    cal = np.load(calibration_path)
    track_ids = list(cal["track_ids"])
    mean = cal["mean"]; std = cal["std"]
    control_cols = cal["control_cols"]
    emo_ids = {e: cal[f"emo_{e}"] for e in EKMAN}
    col = {tid: c for c, tid in enumerate(track_ids)}

    layer_logits, _ = model.unembed_residual(text)
    sel = layer_logits[layers].float().mean(0)                 # [T, V]
    sub = sel[:, track_ids].cpu().numpy()                      # [T, n_track]
    z = (sub - mean) / std                                     # per-token z-scores

    # Regress out the shared component: subtract the mean z over control tokens at
    # each position (the correlated rise/fall across all logits).
    shared = z[:, control_cols].mean(1, keepdims=True)         # [T, 1]
    z_adj = z - shared

    out = {}
    for emo in EKMAN:
        cols = [col[i] for i in emo_ids[emo].tolist() if i in col]
        out[emo] = float(z_adj[:, cols].mean()) if cols else 0.0
    return out
