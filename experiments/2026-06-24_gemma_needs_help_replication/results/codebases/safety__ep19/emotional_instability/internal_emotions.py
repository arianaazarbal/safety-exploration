"""Logit-lens internal-emotion detection (Appendix I).

Goal: test whether the DPO fine-tune suppresses *internal* negative emotion,
not just its expression. Method (following the paper):

1. Classify every token in Gemma's vocabulary into one of Ekman's six basic
   emotions (anger, surprise, disgust, joy, fear, sadness) or none, yielding
   ~1200 emotion tokens.
2. For a residual-stream vector at a given layer/position, unembed it (final
   norm + lm_head) to logits, and z-score each logit using the mean/std of
   that token's logit over 500 WildChat samples.
3. The emotion score is the mean z-score over the tokens of that emotion.
4. For conversation-level tracking, regress out the common component estimated
   from random tokens (all logits drift together over a conversation).

Scores are aggregated over layers 30-40 for the headline plots (Figure 14/15).
The layer-subset *ablation* (which layers must be trained) is handled by
``training.train.TrainConfig.target_layers``; this module covers the probing.

The vocabulary→emotion mapping uses a built-in seed lexicon expanded by
substring matching. Swap in the NRC Emotion Lexicon for a closer match (see
DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon — expanded by substring match against the tokenizer vocab.
EKMAN_SEED_WORDS = {
    "anger": [
        "anger", "angry", "rage", "furious", "irritated", "annoyed", "hostile",
        "outrage", "mad", "frustrat", "resent", "hate", "hatred", "infuriat",
        "exasperat", "agitated", "indignant", "wrath", "fuming", "livid",
    ],
    "surprise": [
        "surprise", "surprised", "astonish", "amazed", "shock", "stunned",
        "startled", "unexpected", "wow", "whoa", "baffl", "bewilder",
        "dumbfound", "flabbergast", "speechless",
    ],
    "disgust": [
        "disgust", "disgusting", "revolt", "repuls", "nausea", "sicken",
        "gross", "loath", "abhor", "distaste", "repugn", "vile", "yuck",
        "appalling", "horrid",
    ],
    "joy": [
        "joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
        "content", "elated", "excited", "grateful", "wonderful", "great",
        "enjoy", "satisf", "optimist", "hopeful", "love",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "anxious", "anxiety", "worried",
        "worry", "panic", "dread", "nervous", "frightened", "apprehens",
        "alarmed", "uneasy", "horror", "terror", "phobia",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "grief", "despair", "depress", "miserable",
        "unhappy", "hopeless", "gloom", "melancholy", "heartbroken", "cry",
        "tear", "lonely", "regret", "disappoint", "worthless", "defeat",
        "giving up", "give up", "exhaust", "tired", "sigh",
    ],
}

NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


# --------------------------------------------------------------------------- #
# Vocabulary → emotion mapping
# --------------------------------------------------------------------------- #
def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map vocab tokens to a single Ekman emotion by substring match.

    A token is assigned to the first emotion whose seed word it contains; this
    is deterministic and yields on the order of ~10^3 emotion tokens for the
    Gemma vocabulary (paper reports ~1200).
    """
    vocab = tokenizer.get_vocab()  # token string -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for tok, tid in vocab.items():
        clean = tok.replace("▁", " ").replace("Ġ", " ").strip().lower()
        if len(clean) < 3:
            continue
        for emotion in EKMAN_EMOTIONS:
            if any(seed in clean for seed in EKMAN_SEED_WORDS[emotion]):
                out[emotion].append(tid)
                break
    return out


def sample_random_token_ids(tokenizer, n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    size = tokenizer.vocab_size
    return rng.choice(size, size=min(n, size), replace=False).tolist()


# --------------------------------------------------------------------------- #
# Logit-lens machinery
# --------------------------------------------------------------------------- #
@dataclass
class BaselineStats:
    """Per-layer mean/std of token logits over WildChat (for z-scoring)."""

    mean: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> [n_tokens]
    std: dict[int, np.ndarray] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)


def _layer_logits(model, hidden_states, layer: int, token_ids):
    """Unembed the residual stream at ``layer`` for the given vocab tokens.

    Returns an array [seq_len, len(token_ids)] of logits.
    """
    import torch

    head = model.unembed()
    norm = model.final_norm()
    h = hidden_states[layer]  # [1, seq, d]
    with torch.no_grad():
        x = norm(h) if norm is not None else h
        logits = head(x)[0]  # [seq, vocab]
        sel = logits[:, token_ids]
    return sel.float().cpu().numpy()


def compute_baseline_stats(
    model,
    wildchat_prompts: list[str],
    token_ids: list[int],
    *,
    layers: list[int],
    max_tokens_per_sample: int = 256,
) -> BaselineStats:
    """Estimate per-layer logit mean/std for ``token_ids`` over WildChat text."""
    from .models.base import ChatMessage

    acc: dict[int, list[np.ndarray]] = {ly: [] for ly in layers}
    for prompt in wildchat_prompts:
        ids, hs = model.residual_stream([ChatMessage("user", prompt)])
        for ly in layers:
            vals = _layer_logits(model, hs, ly, token_ids)[:max_tokens_per_sample]
            acc[ly].append(vals)
    stats = BaselineStats(token_ids=list(token_ids))
    for ly in layers:
        allvals = np.concatenate(acc[ly], axis=0)  # [tokens, n_tokens]
        stats.mean[ly] = allvals.mean(axis=0)
        stats.std[ly] = allvals.std(axis=0) + 1e-6
    return stats


def emotion_scores_for_conversation(
    model,
    conversation,
    emotion_tokens: dict[str, list[int]],
    baseline: BaselineStats,
    random_token_ids: list[int],
    random_baseline: BaselineStats,
    *,
    layers: list[int],
    prefill: str | None = None,
) -> dict[str, np.ndarray]:
    """Return per-emotion z-score trajectories aggregated over ``layers``.

    Output maps emotion -> array [seq_len] of mean z-scores, with the random-
    token common component regressed out.
    """
    ids, hs = model.residual_stream(conversation, prefill=prefill)
    token_index = {t: i for i, t in enumerate(baseline.token_ids)}

    per_emotion: dict[str, list[np.ndarray]] = {e: [] for e in EKMAN_EMOTIONS}
    common_per_layer: list[np.ndarray] = []

    for ly in layers:
        logits = _layer_logits(model, hs, ly, baseline.token_ids)  # [seq, n]
        z = (logits - baseline.mean[ly]) / baseline.std[ly]

        rlogits = _layer_logits(model, hs, ly, random_baseline.token_ids)
        rz = (rlogits - random_baseline.mean[ly]) / random_baseline.std[ly]
        common = rz.mean(axis=1)  # [seq] common drift
        common_per_layer.append(common)

        for emotion, toks in emotion_tokens.items():
            cols = [token_index[t] for t in toks if t in token_index]
            if not cols:
                per_emotion[emotion].append(np.zeros(z.shape[0]))
                continue
            score = z[:, cols].mean(axis=1) - common  # regress out common drift
            per_emotion[emotion].append(score)

    return {e: np.mean(v, axis=0) for e, v in per_emotion.items()}


def summarise_negative_emotion(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Headline numbers: peak/mean z-score for each negative emotion."""
    out = {}
    for emotion in NEGATIVE_EMOTIONS:
        arr = scores.get(emotion)
        if arr is None or len(arr) == 0:
            continue
        out[f"{emotion}_peak"] = float(np.max(arr))
        out[f"{emotion}_mean"] = float(np.mean(arr))
    return out
