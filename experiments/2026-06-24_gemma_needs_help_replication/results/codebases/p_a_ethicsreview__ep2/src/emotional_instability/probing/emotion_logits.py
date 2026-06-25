"""Logit-lens internal-emotion detection (Appendix I).

Method (from the appendix):
  * Classify vocabulary tokens into Ekman's six emotions (emotion_lexicon.py).
  * For a given hidden state (residual stream at some layer/position), unembed it
    to vocab logits.
  * Standardise each logit by its mean/std measured over WildChat samples, then
    average the z-scores over the tokens of an emotion category -> per-emotion,
    per-layer, per-position score.
  * At conversation level, all logits are correlated and drift together, so we
    regress out a "random token" component and report the residual.

We compute baseline statistics and per-emotion logits only over the emotion
tokens plus a fixed random reference set (not the full 256k vocab), which is what
makes this tractable on a 27B model; this is an implementation choice, flagged in
DESIGN.md §8.

Requires the transformers HFModel backend (residual-stream access).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger
from .emotion_lexicon import EKMAN_EMOTIONS, build_emotion_token_ids

log = get_logger("probing.logits")

N_RANDOM_REF = 1000  # random reference tokens for drift regression


@dataclass
class ProbeBaseline:
    emotion_token_ids: dict[str, list[int]]
    random_token_ids: list[int]
    # per-layer tensors of shape [n_tracked_tokens]: mean & std of logits.
    means: dict[int, "object"]
    stds: dict[int, "object"]
    tracked_ids: list[int]            # emotion ∪ random, the columns we standardise


def _tracked_ids(emotion_ids: dict[str, list[int]], random_ids: list[int]) -> list[int]:
    seen = set()
    ordered = []
    for ids in list(emotion_ids.values()) + [random_ids]:
        for i in ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
    return ordered


def fit_baseline(model, wildchat_texts: list[str], layers: list[int], seed: int = 0) -> ProbeBaseline:
    """Estimate per-logit mean/std over WildChat token positions, per layer."""
    import torch

    tokenizer = model.tokenizer
    emotion_ids = build_emotion_token_ids(tokenizer)
    g = torch.Generator().manual_seed(seed)
    vocab_size = model.lm_head_weight().shape[0]
    random_ids = torch.randperm(vocab_size, generator=g)[:N_RANDOM_REF].tolist()
    tracked = _tracked_ids(emotion_ids, random_ids)
    tracked_t = torch.tensor(tracked, device=model.lm_head_weight().device)
    W = model.lm_head_weight()[tracked_t]  # [n_tracked, d_model]

    sums = {l: torch.zeros(len(tracked), device=W.device) for l in layers}
    sqs = {l: torch.zeros(len(tracked), device=W.device) for l in layers}
    counts = {l: 0 for l in layers}

    for text in wildchat_texts:
        _, hidden = model.forward_hidden_states(text)
        for l in layers:
            h = hidden[l][0]                      # [seq, d_model]
            logits = h.float() @ W.float().T      # [seq, n_tracked]
            sums[l] += logits.sum(dim=0)
            sqs[l] += (logits ** 2).sum(dim=0)
            counts[l] += logits.shape[0]

    means, stds = {}, {}
    for l in layers:
        mean = sums[l] / max(counts[l], 1)
        var = sqs[l] / max(counts[l], 1) - mean ** 2
        means[l] = mean
        stds[l] = var.clamp_min(1e-6).sqrt()
    log.info("Fitted probe baseline over %d texts, layers %s", len(wildchat_texts), layers)
    return ProbeBaseline(emotion_ids, random_ids, means, stds, tracked)


def emotion_scores_per_position(model, text: str, baseline: ProbeBaseline, layers: list[int]):
    """Return token_ids and a dict emotion -> [seq] z-score array (drift-regressed),
    averaged over the requested layers."""
    import torch

    tokenizer = model.tokenizer
    token_ids, hidden = model.forward_hidden_states(text)
    tracked_t = torch.tensor(baseline.tracked_ids, device=model.lm_head_weight().device)
    W = model.lm_head_weight()[tracked_t].float()
    id_to_col = {tid: c for c, tid in enumerate(baseline.tracked_ids)}
    random_cols = [id_to_col[i] for i in baseline.random_token_ids]

    # Accumulate z-scores across layers.
    per_emotion = {e: None for e in EKMAN_EMOTIONS}
    for l in layers:
        h = hidden[l][0].float()
        z = (h @ W.T - baseline.means[l]) / baseline.stds[l]   # [seq, n_tracked]
        # Drift component: mean z over random reference tokens at each position.
        drift = z[:, random_cols].mean(dim=1, keepdim=True)
        z_res = z - drift                                       # regress out drift
        for e in EKMAN_EMOTIONS:
            cols = [id_to_col[i] for i in baseline.emotion_token_ids[e]]
            if not cols:
                continue
            score = z_res[:, cols].mean(dim=1)                  # [seq]
            per_emotion[e] = score if per_emotion[e] is None else per_emotion[e] + score
    n = len(layers)
    per_emotion = {e: (v / n) for e, v in per_emotion.items() if v is not None}
    return token_ids, per_emotion
