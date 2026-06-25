"""Logit-based internal-emotion detection (Appendix I).

We detect internal emotions by unembedding the residual stream and aggregating
over emotion-related tokens:

  1. Classify the vocabulary into Ekman's 6 basic emotions (anger, surprise,
     disgust, joy, fear, sadness) via seed-word stems -> ~emotion token sets.
  2. For a given layer, unembed the residual stream to per-token logits, then
     standardise each emotion token's logit using its mean/std over a baseline
     corpus (500 WildChat samples).
  3. Average the z-scores within an emotion category. Because all logits drift
     together over a conversation, we regress out the mean over random baseline
     tokens to isolate emotion-specific signal.

This avoids training linear probes (no probe data needed), matching the paper's
rationale. The vocabulary classification is necessarily approximate (the paper's
exact dictionary is not published) -- documented in DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..llm.gemma_local import GemmaModel

# Seed stems per Ekman emotion. Vocabulary tokens whose normalised form starts
# with / equals any stem are assigned to that emotion.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "mad",
              "hostil", "outrage", "resent", "frustrat", "wrath", "hate"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock", "startl",
                 "unexpected", "stun", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "gross", "sicken", "loath",
                "contempt", "yuck"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "pleased",
            "excited", "content", "grateful", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worry", "worried",
             "terrif", "panic", "dread", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief", "depress",
                "gloom", "unhappy", "cry", "tears", "lonely", "worthless"],
}


@dataclass
class EmotionTokens:
    by_emotion: dict[str, list[int]]
    random_ids: list[int]

    @property
    def tracked(self) -> list[int]:
        ids = set(self.random_ids)
        for v in self.by_emotion.values():
            ids.update(v)
        return sorted(ids)


def build_emotion_tokens(model: GemmaModel, *, n_random: int = 1000,
                         seed: int = 0) -> EmotionTokens:
    tok = model.tokenizer
    vocab_size = len(tok)
    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    assigned: set[int] = set()
    for tid in range(vocab_size):
        word = tok.decode([tid]).strip().lower()
        if not word or not word.isalpha():
            continue
        for emo, stems in EKMAN_SEEDS.items():
            if any(word.startswith(s) for s in stems):
                by_emotion[emo].append(tid)
                assigned.add(tid)
                break
    rng = np.random.default_rng(seed)
    pool = [t for t in range(vocab_size) if t not in assigned]
    random_ids = sorted(rng.choice(pool, size=min(n_random, len(pool)), replace=False).tolist())
    return EmotionTokens(by_emotion=by_emotion, random_ids=random_ids)


@dataclass
class Baseline:
    layers: list[int]
    ids: list[int]                  # tracked token ids, in column order
    mean: np.ndarray                # (n_layers, n_ids)
    std: np.ndarray                 # (n_layers, n_ids)


def _layer_logits(model: GemmaModel, text: str, layers: list[int],
                  ids: torch.Tensor) -> np.ndarray:
    """Per-position logits at ``ids`` for each requested layer.

    Returns array of shape (n_layers, seq_len, n_ids).
    """
    hs = model.residual_stream(text)                 # (L+1, T, d)
    W = model.unembed.index_select(0, ids)           # (n_ids, d)
    out = []
    for L in layers:
        logits = hs[L].to(W.dtype) @ W.T             # (T, n_ids)
        out.append(logits.float().cpu().numpy())
    return np.stack(out, axis=0)


def fit_baseline(model: GemmaModel, texts: list[str], emo_tokens: EmotionTokens,
                 layers: list[int]) -> Baseline:
    ids = emo_tokens.tracked
    ids_t = torch.tensor(ids, device=model.device)
    sums = np.zeros((len(layers), len(ids)))
    sqs = np.zeros((len(layers), len(ids)))
    count = 0
    for text in texts:
        ll = _layer_logits(model, text, layers, ids_t)   # (Ln, T, n_ids)
        sums += ll.sum(axis=1)
        sqs += (ll ** 2).sum(axis=1)
        count += ll.shape[1]
    mean = sums / max(1, count)
    var = np.maximum(sqs / max(1, count) - mean ** 2, 1e-6)
    return Baseline(layers=layers, ids=ids, mean=mean, std=np.sqrt(var))


def emotion_zscores(model: GemmaModel, text: str, emo_tokens: EmotionTokens,
                    baseline: Baseline) -> dict[str, np.ndarray]:
    """Per-position, emotion-category z-scores, averaged over baseline layers.

    Returns {emotion: array of shape (seq_len,)} after regressing out the mean
    z-score over random baseline tokens at each position.
    """
    ids = baseline.ids
    id_to_col = {t: c for c, t in enumerate(ids)}
    ids_t = torch.tensor(ids, device=model.device)
    ll = _layer_logits(model, text, baseline.layers, ids_t)       # (Ln, T, n_ids)
    z = (ll - baseline.mean[:, None, :]) / baseline.std[:, None, :]
    z = z.mean(axis=0)                                            # (T, n_ids) avg over layers

    rand_cols = [id_to_col[t] for t in emo_tokens.random_ids if t in id_to_col]
    rand_mean = z[:, rand_cols].mean(axis=1)                     # (T,)

    out = {}
    for emo, toks in emo_tokens.by_emotion.items():
        cols = [id_to_col[t] for t in toks if t in id_to_col]
        if not cols:
            out[emo] = np.zeros(z.shape[0])
            continue
        out[emo] = z[:, cols].mean(axis=1) - rand_mean           # regress out drift
    return out
