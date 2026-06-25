"""Logit-based internal emotion detection (Appendix I).

Method (from Appendix I):
  1. Classify vocabulary tokens into Ekman's six emotions (emotion_lexicon).
  2. For a piece of text, unembed the residual stream at each layer (apply the
     model's output head to each layer's hidden state) to get per-token,
     per-vocab logits at that layer.
  3. Standardise each vocab logit by its mean/std computed over 500 WildChat
     samples (z-score).
  4. Average z-scores over the tokens in an emotion category -> a per-layer,
     per-position emotion score.
  5. For conversation-level scores, regress out the common component shared with
     random tokens (all logits are correlated and drift over a conversation).

We compare the vanilla instruct model with the DPO finetune to show internal
negative emotions are suppressed, not just expression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnh.config import ARTIFACT_DIR
from gnh.internal.emotion_lexicon import EKMAN_EMOTIONS, build_emotion_token_ids
from gnh.prompts.wildchat import load_wildchat_prompts


@dataclass
class EmotionStats:
    """Per-layer mean/std of each vocab logit, for z-scoring."""

    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]


def _layer_logits(backend, text: str) -> np.ndarray:
    """Return [n_layers, seq, vocab] logits from unembedding each layer's
    residual stream. Uses the model's tied/untied output head."""

    import torch

    hidden_states, _ = backend.residual_stream(text)
    model = backend.model
    # Resolve the unembedding matrix (lm_head); handle PEFT-wrapped models.
    base = getattr(model, "base_model", model)
    lm_head = base.get_output_embeddings()
    W = lm_head.weight  # [vocab, d_model]

    out = []
    with torch.no_grad():
        for h in hidden_states:                 # each [1, seq, d_model]
            logits = (h[0] @ W.T).float().cpu().numpy()   # [seq, vocab]
            out.append(logits)
    return np.stack(out, axis=0)                 # [n_layers, seq, vocab]


def fit_baseline_stats(backend, n_samples: int = 500, seed: int = 0) -> EmotionStats:
    """Collect per-layer logit mean/std over WildChat samples for z-scoring."""

    prompts = load_wildchat_prompts(n=min(20, n_samples), seed=seed)
    # Repeat/sample to reach n_samples short texts.
    texts = [prompts[i % len(prompts)] for i in range(n_samples)]

    sums = sumsq = count = None
    for t in texts:
        lg = _layer_logits(backend, t)          # [L, seq, V]
        flat = lg.reshape(lg.shape[0], -1, lg.shape[2])  # [L, seq, V]
        s = flat.sum(axis=1)                     # [L, V]
        sq = (flat**2).sum(axis=1)
        c = flat.shape[1]
        if sums is None:
            sums, sumsq, count = s, sq, c
        else:
            sums, sumsq, count = sums + s, sumsq + sq, count + c
    mean = sums / count
    var = np.maximum(sumsq / count - mean**2, 1e-6)
    return EmotionStats(mean=mean, std=np.sqrt(var))


def emotion_scores(
    backend,
    text: str,
    stats: EmotionStats,
    emotion_token_ids: dict[str, list[int]],
    *,
    regress_random: bool = True,
    n_random: int = 500,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return {emotion: [n_layers, seq]} z-scored emotion activation.

    If ``regress_random`` is set, subtract the mean z-score over a random token
    set at each (layer, position) to remove the shared/correlated component.
    """

    lg = _layer_logits(backend, text)            # [L, seq, V]
    z = (lg - stats.mean[:, None, :]) / stats.std[:, None, :]   # [L, seq, V]

    rng = np.random.default_rng(seed)
    random_ids = rng.choice(lg.shape[2], size=min(n_random, lg.shape[2]), replace=False)
    baseline = z[:, :, random_ids].mean(axis=2) if regress_random else 0.0  # [L, seq]

    out = {}
    for emotion in EKMAN_EMOTIONS:
        ids = emotion_token_ids.get(emotion, [])
        if not ids:
            out[emotion] = np.zeros(lg.shape[:2])
            continue
        score = z[:, :, ids].mean(axis=2)        # [L, seq]
        out[emotion] = score - baseline
    return out


def compare_models_on_conversation(
    vanilla_backend,
    dpo_backend,
    conversation_text: str,
    *,
    layers=(30, 40),
    window: int = 400,
) -> dict:
    """Compare running-average emotion scores (layers averaged over ``layers``)
    along ``conversation_text`` for vanilla vs DPO (Figure 14)."""

    em_ids_v = build_emotion_token_ids(vanilla_backend.tokenizer)
    stats_v = fit_baseline_stats(vanilla_backend)
    stats_d = fit_baseline_stats(dpo_backend)

    def running(backend, stats):
        scores = emotion_scores(backend, conversation_text, stats, em_ids_v)
        lo, hi = layers
        out = {}
        for emo, arr in scores.items():
            layer_avg = arr[lo:hi].mean(axis=0)             # [seq]
            # Running average over `window` tokens.
            kernel = np.ones(min(window, len(layer_avg))) / min(window, len(layer_avg))
            out[emo] = np.convolve(layer_avg, kernel, mode="valid").tolist()
        return out

    result = {"vanilla": running(vanilla_backend, stats_v),
              "dpo": running(dpo_backend, stats_d)}
    (ARTIFACT_DIR / "internal_emotion_trajectory.json").write_text(json.dumps(result))
    return result
