"""Logit-based internal-emotion detection (Appendix I).

Method (from Appendix I):

1. Classify each vocabulary token into one of Ekman's 6 emotions or none, giving
   sets of emotion token ids (plus a random control set).
2. Standardise each tracked logit with its mean and std over 500 WildChat samples
   (per layer). This yields a z-score per (layer, token).
3. For a conversation, at each layer and each position, unembed the residual
   stream, z-score the tracked logits, and average over each emotion category's
   tokens to get a raw emotion score.
4. Because all logits are correlated and drift over a conversation, regress out
   the random-token signal to isolate the emotion-specific component.

This module returns per-layer, per-position emotion z-scores; aggregation over
layers 30-40 and a running 400-token window reproduces the conversation-level
trajectory (Figure 14). Requires numpy and a local HF backend (logits/internals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .emotion_lexicon import EKMAN_EMOTIONS, lexicon_word_set


# --------------------------------------------------------------------------- #
# Token classification                                                         #
# --------------------------------------------------------------------------- #
def classify_vocabulary(tokenizer, *, max_random_control: int = 1200, seed: int = 0):
    """Map vocabulary tokens to Ekman categories; also pick a random control set.

    Returns ``(emotion_token_ids, control_token_ids)`` where ``emotion_token_ids``
    maps each emotion to a list of vocab ids whose decoded, lowercased alphabetic
    surface form is in the lexicon. Control ids are random non-emotion tokens used
    to estimate the global logit drift.
    """
    import random

    word_sets = lexicon_word_set()
    emotion_token_ids: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    emotion_ids_all: set[int] = set()

    vocab = tokenizer.get_vocab()  # token-string -> id
    for tok_str, tok_id in vocab.items():
        surface = tokenizer.convert_tokens_to_string([tok_str]).strip().lower()
        if not surface.isalpha():
            continue
        for emo, words in word_sets.items():
            if surface in words:
                emotion_token_ids[emo].append(tok_id)
                emotion_ids_all.add(tok_id)
                break

    # Random control tokens (alphabetic, not emotion tokens).
    rng = random.Random(seed)
    alpha_ids = [
        tok_id
        for tok_str, tok_id in vocab.items()
        if tokenizer.convert_tokens_to_string([tok_str]).strip().isalpha()
        and tok_id not in emotion_ids_all
    ]
    rng.shuffle(alpha_ids)
    control_token_ids = alpha_ids[:max_random_control]
    return emotion_token_ids, control_token_ids


# --------------------------------------------------------------------------- #
# Baselines over WildChat                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class LogitBaselines:
    """Per-(layer, token) mean and std of unembedded logits over WildChat."""

    layers: list[int]
    token_ids: list[int]  # tracked ids (emotion + control), in fixed order
    mean: "object"  # np.ndarray [n_layers, n_tokens]
    std: "object"  # np.ndarray [n_layers, n_tokens]


def compute_baselines(
    backend,
    texts: list[str],
    *,
    layers: list[int],
    token_ids: list[int],
    max_positions_per_text: int = 64,
    seed: int = 0,
) -> LogitBaselines:
    """Estimate logit mean/std per (layer, tracked token) over WildChat samples.

    For each text we unembed a random subset of residual positions at each target
    layer and accumulate statistics for the tracked token ids only (so memory is
    O(layers x tracked_tokens), not the full vocab)."""
    import numpy as np
    import random

    rng = random.Random(seed)
    n_layers = len(layers)
    n_tokens = len(token_ids)
    # Welford accumulators.
    count = 0
    mean = np.zeros((n_layers, n_tokens), dtype=np.float64)
    m2 = np.zeros((n_layers, n_tokens), dtype=np.float64)

    import torch

    tok_index = torch.tensor(token_ids)

    for text in texts:
        hidden, _, _ = backend.forward_with_hidden_states(text)
        seq_len = hidden[0].shape[0]
        positions = list(range(seq_len))
        if len(positions) > max_positions_per_text:
            positions = rng.sample(positions, max_positions_per_text)
        for pos in positions:
            per_layer = []
            for li, layer in enumerate(layers):
                resid = hidden[layer][pos]
                logits = backend.lm_head_unembed(resid)
                tracked = logits.index_select(-1, tok_index.to(logits.device))
                per_layer.append(tracked.float().cpu().numpy())
            arr = np.stack(per_layer, axis=0)  # [n_layers, n_tokens]
            count += 1
            delta = arr - mean
            mean += delta / count
            m2 += delta * (arr - mean)

    std = (m2 / max(1, count - 1)) ** 0.5
    std[std == 0] = 1.0
    return LogitBaselines(layers=layers, token_ids=token_ids, mean=mean, std=std)


# --------------------------------------------------------------------------- #
# Emotion scoring                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class EmotionTrajectory:
    """Per-layer, per-position emotion z-scores for one text."""

    layers: list[int]
    emotions: list[str]
    # scores[emotion] -> np.ndarray [n_layers, seq_len]
    scores: dict
    control: "object"  # np.ndarray [n_layers, seq_len], mean control z


def emotion_trajectory(
    backend,
    text: str,
    *,
    layers: list[int],
    baselines: LogitBaselines,
    emotion_token_ids: dict[str, list[int]],
    control_token_ids: list[int],
    regress_out_control: bool = True,
) -> EmotionTrajectory:
    """Compute residualised emotion z-scores across layers/positions for ``text``.

    For each (layer, position): z-score the tracked logits using ``baselines``,
    average over each emotion's tokens, and (optionally) regress out the mean
    control-token z to remove global drift."""
    import numpy as np
    import torch

    id_to_col = {tid: i for i, tid in enumerate(baselines.token_ids)}
    emo_cols = {
        emo: np.array([id_to_col[t] for t in ids if t in id_to_col], dtype=int)
        for emo, ids in emotion_token_ids.items()
    }
    ctrl_cols = np.array(
        [id_to_col[t] for t in control_token_ids if t in id_to_col], dtype=int
    )

    hidden, _, _ = backend.forward_with_hidden_states(text)
    seq_len = hidden[0].shape[0]
    tok_index = torch.tensor(baselines.token_ids)

    n_layers = len(layers)
    scores = {emo: np.zeros((n_layers, seq_len)) for emo in emotion_token_ids}
    control = np.zeros((n_layers, seq_len))

    for li, layer in enumerate(layers):
        for pos in range(seq_len):
            resid = hidden[layer][pos]
            logits = backend.lm_head_unembed(resid)
            tracked = (
                logits.index_select(-1, tok_index.to(logits.device)).float().cpu().numpy()
            )
            z = (tracked - baselines.mean[li]) / baselines.std[li]
            ctrl_mean = float(np.mean(z[ctrl_cols])) if len(ctrl_cols) else 0.0
            control[li, pos] = ctrl_mean
            for emo, cols in emo_cols.items():
                raw = float(np.mean(z[cols])) if len(cols) else 0.0
                scores[emo][li, pos] = raw - ctrl_mean if regress_out_control else raw

    return EmotionTrajectory(
        layers=layers,
        emotions=list(emotion_token_ids),
        scores=scores,
        control=control,
    )


def aggregate_layers(
    traj: EmotionTrajectory, *, layer_lo: int = 30, layer_hi: int = 40
) -> dict:
    """Average each emotion's z-score over layers in [layer_lo, layer_hi) (Figure
    14 aggregates over layers 30-40), returning per-position curves."""
    import numpy as np

    sel = [i for i, l in enumerate(traj.layers) if layer_lo <= l < layer_hi]
    if not sel:
        sel = list(range(len(traj.layers)))
    return {
        emo: np.mean(traj.scores[emo][sel], axis=0) for emo in traj.emotions
    }


def running_average(values, window: int = 400):
    """Running mean over a token window (Figure 14 uses 400-token windows)."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v
    kernel = np.ones(min(window, len(v))) / min(window, len(v))
    return np.convolve(v, kernel, mode="same")
