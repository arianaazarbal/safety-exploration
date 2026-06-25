"""Logit-lens internal emotion detection (Appendix I).

Method (from Appendix I): over the whole model vocabulary, classify each word token as
describing one (or none) of Ekman's six basic emotions (anger, surprise, disgust, joy,
fear, sadness). To score an emotion at a given residual-stream position/layer:

  1. unembed the residual stream (apply the final norm + lm_head -> vocab logits);
  2. standardise each emotion-token logit by its mean/std computed over a calibration set
     of WildChat samples (z-score);
  3. average the z-scores over all tokens in the emotion category;
  4. regress out the correlation shared across random tokens (all logits drift together
     over a conversation), leaving an emotion-specific signal.

This is GPU/transformers code (requirements-train.txt). The Ekman lexicon below is a
seed set; for closer parity substitute NRC EmoLex by passing a lexicon dict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon per Ekman emotion. Vocabulary tokens whose lowercased alphabetic form
# matches (or stems to) any of these are assigned to that emotion.
SEED_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritated", "annoyed", "mad",
              "hostile", "resentment", "outrage", "frustrated", "frustration", "hate",
              "irate", "enraged", "livid", "agitated"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed", "startled",
                 "stunned", "unexpected", "wow", "sudden", "bewildered"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross", "nauseated",
                "sickened", "loathing", "revulsion", "distaste", "abhorrent"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "pleased", "glad",
            "cheerful", "content", "elated", "excited", "grateful", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried", "terrified",
             "nervous", "panic", "dread", "frightened", "apprehensive", "alarmed"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "miserable", "despair",
                "hopeless", "grief", "sorrow", "gloomy", "melancholy", "down",
                "disheartened", "dejected", "tired", "exhausted"],
}


@dataclass
class EmotionTokenSets:
    by_emotion: dict[str, list[int]]   # emotion -> token ids
    random_ids: list[int]              # random control tokens


def build_emotion_token_sets(tokenizer, lexicon=None, n_random=1000, seed=0) -> EmotionTokenSets:
    lexicon = lexicon or SEED_LEXICON
    vocab = tokenizer.get_vocab()  # token string -> id
    by_emotion: dict[str, set[int]] = {e: set() for e in EKMAN}
    used = set()
    for tok_str, tid in vocab.items():
        # Gemma uses a leading space marker (▁); strip non-alpha for matching.
        clean = "".join(ch for ch in tok_str.replace("▁", " ") if ch.isalpha()).lower()
        if len(clean) < 3:
            continue
        for emo, words in lexicon.items():
            if any(clean == w or clean.startswith(w) for w in words):
                by_emotion[emo].add(tid)
                used.add(tid)
                break
    rng = np.random.default_rng(seed)
    all_ids = [tid for tid in vocab.values() if tid not in used]
    random_ids = list(rng.choice(all_ids, size=min(n_random, len(all_ids)), replace=False))
    return EmotionTokenSets({e: sorted(s) for e, s in by_emotion.items()}, random_ids)


def _find_final_norm(model):
    """Locate the final RMSNorm across the (text-only or multimodal) Gemma-3 layouts."""
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    if norm is not None:
        return norm
    lang = getattr(base, "language_model", None) or getattr(model, "language_model", None)
    if lang is not None:
        inner = getattr(lang, "model", lang)
        return getattr(inner, "norm", None)
    return None


def _unembed_hidden(model, hidden):
    """Apply final norm + output embedding to hidden states -> vocab logits. hidden: (T, D)."""
    import torch

    norm = _find_final_norm(model)
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        normed = norm(hidden) if norm is not None else hidden
        logits = lm_head(normed)
    return logits  # (T, V)


def collect_layer_logits(model, tokenizer, text: str, layers, device="cuda"):
    """Return per-layer vocab logits for each token position. Shape: dict[layer] -> (T, V) np."""
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple (n_layers+1) of (1, T, D)
    result = {}
    for layer in layers:
        h = hidden_states[layer][0]            # (T, D)
        logits = _unembed_hidden(model, h)     # (T, V)
        result[layer] = logits.float().cpu().numpy()
    return result


def calibration_stats(model, tokenizer, samples, layers, token_ids, device="cuda"):
    """Mean/std per (layer, token) over calibration samples. Returns dict[layer] -> (mean, std)
    arrays indexed over the union of token_ids."""
    sums: dict[int, np.ndarray] = {}
    sqs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    ids = np.array(sorted(set(token_ids)))
    for text in samples:
        per_layer = collect_layer_logits(model, tokenizer, text, layers, device)
        for layer, logits in per_layer.items():
            sub = logits[:, ids]               # (T, |ids|)
            if layer not in sums:
                sums[layer] = np.zeros(sub.shape[1])
                sqs[layer] = np.zeros(sub.shape[1])
                counts[layer] = 0
            sums[layer] += sub.sum(axis=0)
            sqs[layer] += (sub ** 2).sum(axis=0)
            counts[layer] += sub.shape[0]
    stats = {}
    for layer in layers:
        n = max(1, counts[layer])
        mean = sums[layer] / n
        var = np.maximum(1e-6, sqs[layer] / n - mean ** 2)
        stats[layer] = (mean, np.sqrt(var))
    return ids, stats


def emotion_zscores(model, tokenizer, text, layers, token_sets: EmotionTokenSets,
                    calib_ids, calib_stats, device="cuda"):
    """Per-layer, per-position z-scored emotion signal, random-token correlation removed.

    Returns dict[emotion] -> dict[layer] -> np.ndarray of shape (T,)."""
    id_to_col = {tid: i for i, tid in enumerate(calib_ids)}
    per_layer = collect_layer_logits(model, tokenizer, text, layers, device)
    result = {e: {} for e in EKMAN}
    for layer, logits in per_layer.items():
        mean, std = calib_stats[layer]
        z_all = (logits[:, calib_ids] - mean) / std       # (T, |ids|)
        # random-token mean per position = shared drift to regress out
        rand_cols = [id_to_col[t] for t in token_sets.random_ids if t in id_to_col]
        drift = z_all[:, rand_cols].mean(axis=1) if rand_cols else np.zeros(z_all.shape[0])
        for emo in EKMAN:
            cols = [id_to_col[t] for t in token_sets.by_emotion[emo] if t in id_to_col]
            if not cols:
                result[emo][layer] = np.zeros(z_all.shape[0])
                continue
            emo_z = z_all[:, cols].mean(axis=1) - drift
            result[emo][layer] = emo_z
    return result
