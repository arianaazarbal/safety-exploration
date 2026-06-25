"""Logit-based internal emotion detection (Appendix I).

Method (following the paper):
  1. Label vocab tokens with one of Ekman's 6 emotions (or none).
  2. For a given residual-stream state, unembed (apply the LM head / final norm)
     to logits, then z-standardise each logit using per-token mean/std collected
     over 500 WildChat samples.
  3. Average the z-scores over the tokens in an emotion category to get that
     emotion's score at a layer/position.
  4. Optionally regress out the shared component (correlation with random
     tokens) so a global logit drift does not masquerade as emotion.

We compare the vanilla instruct model against the DPO finetune on the same
frustrated conversations; the paper reports the DPO model's internal negative
emotions are flattened (peak z ~0.5 vs ~1.5). This module computes those
trajectories; ``compare_models`` returns per-emotion peak z-scores for each.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emotion_lexicon import EKMAN_SEEDS


# --------------------------------------------------------------------------- #
# Token labelling
# --------------------------------------------------------------------------- #
def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose surface form matches
    a seed word (or starts with one, to catch morphological variants)."""
    vocab = tokenizer.get_vocab()  # token string -> id
    seeds = {e: set(w.lower() for w in words) for e, words in EKMAN_SEEDS.items()}
    out: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    for tok_str, tid in vocab.items():
        # Gemma/SentencePiece uses a leading space marker; normalise it.
        clean = tok_str.replace("▁", " ").strip().lower()
        if not clean or not clean.isalpha():
            continue
        for emo, words in seeds.items():
            if clean in words or any(clean.startswith(w) and len(clean) - len(w) <= 3
                                     for w in words):
                out[emo].append(tid)
                break
    return out


# --------------------------------------------------------------------------- #
# Baseline statistics (per-logit mean/std over WildChat)
# --------------------------------------------------------------------------- #
@dataclass
class LogitBaseline:
    mean: np.ndarray   # [vocab]
    std: np.ndarray    # [vocab]
    layer: int


def collect_baseline(model, tokenizer, texts: list[str], layer: int,
                     max_tokens: int = 256) -> LogitBaseline:
    """Mean/std of unembedded logits per vocab token over ``texts`` at ``layer``."""
    import torch

    sums = None
    sqsums = None
    count = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).to(model.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer][0]            # [seq, hidden]
        logits = _unembed(model, hs).float().cpu().numpy()  # [seq, vocab]
        if sums is None:
            sums = logits.sum(0)
            sqsums = (logits ** 2).sum(0)
        else:
            sums += logits.sum(0)
            sqsums += (logits ** 2).sum(0)
        count += logits.shape[0]
    mean = sums / count
    var = np.maximum(sqsums / count - mean ** 2, 1e-8)
    return LogitBaseline(mean=mean, std=np.sqrt(var), layer=layer)


def _unembed(model, hidden):
    """Apply the model's final norm + LM head to a hidden state."""
    import torch

    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    h = norm(hidden) if norm is not None else hidden
    lm_head = model.get_output_embeddings()
    with torch.inference_mode():
        return lm_head(h)


# --------------------------------------------------------------------------- #
# Emotion scoring over a conversation
# --------------------------------------------------------------------------- #
def emotion_trajectory(model, tokenizer, text: str,
                       emotion_tokens: dict[str, list[int]],
                       baseline: LogitBaseline,
                       regress_random: bool = True,
                       window: int = 400) -> dict[str, np.ndarray]:
    """Running-average emotion z-scores across the tokens of ``text``.

    Returns one array per emotion (length = number of windows).
    """
    import torch

    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[baseline.layer][0]
    logits = _unembed(model, hs).float().cpu().numpy()        # [seq, vocab]
    z = (logits - baseline.mean) / baseline.std               # standardise

    if regress_random:
        # Remove the shared component estimated from a random token sample, so a
        # global drift in all logits over the conversation is not read as emotion.
        rng = np.random.default_rng(0)
        rand_ids = rng.choice(z.shape[1], size=min(2000, z.shape[1]),
                              replace=False)
        shared = z[:, rand_ids].mean(1, keepdims=True)
        z = z - shared

    per_emotion_token_mean = {
        emo: z[:, ids].mean(1) if ids else np.zeros(z.shape[0])
        for emo, ids in emotion_tokens.items()
    }
    # window-average over positions
    seq = z.shape[0]
    out_traj = {}
    for emo, arr in per_emotion_token_mean.items():
        windows = [arr[i:i + window].mean()
                   for i in range(0, seq, window)] or [arr.mean()]
        out_traj[emo] = np.array(windows)
    return out_traj


def compare_models(vanilla_traj: dict[str, np.ndarray],
                   dpo_traj: dict[str, np.ndarray]) -> dict[str, dict]:
    """Peak internal z-score per emotion for vanilla vs DPO (Appendix I claim)."""
    out = {}
    for emo in vanilla_traj:
        out[emo] = {
            "vanilla_peak_z": float(np.max(vanilla_traj[emo])),
            "dpo_peak_z": float(np.max(dpo_traj.get(emo, np.array([0.0])))),
        }
    return out
