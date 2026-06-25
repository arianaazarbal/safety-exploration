"""Logit-based internal-emotion detection (paper Appendix I).

Goal: test whether the DPO intervention suppresses *internal* negative emotion,
not just externalised text. Method (open-weight Gemma only):

  1. Classify every token in the Gemma vocabulary into one of Ekman's 6 basic
     emotions (anger, surprise, disgust, joy, fear, sadness) or none, yielding
     ~1200 emotion tokens (~200 per category).
  2. For a given residual-stream activation, unembed (apply the LM head) to get a
     logit per vocab token, then z-standardise each logit using its mean/std over
     500 WildChat samples.
  3. An emotion's score = mean z-score over that category's tokens. To remove the
     global "all logits rise/fall together" component, regress out the mean
     z-score of a random token set.
  4. Compare scores across the conversation, and across layers, for the vanilla vs
     DPO model — the paper finds the DPO model has materially lower internal
     negative-emotion scores even on highly frustrated responses.

This module provides the building blocks; `compute_emotion_trajectory` runs the
full per-token/per-layer trajectory for one conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicons; expanded by morphological/substring matching over the vocab.
_EMOTION_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "outrage", "resent", "frustrat", "hate"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "unexpected",
                 "stun", "wow"],
    "disgust": ["disgust", "revolt", "gross", "nausea", "repuls", "sick",
                "contempt", "loath"],
    "joy": ["joy", "happy", "delight", "glad", "pleased", "cheer", "content",
            "excited", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "anxious", "worry", "worried", "panic",
             "terrif", "dread", "nervous", "apprehens"],
    "sadness": ["sad", "depress", "despair", "hopeless", "miser", "grief",
                "sorrow", "unhappy", "cry", "tear", "lonely", "worthless"],
}


@dataclass
class EmotionLexicon:
    token_ids: dict[str, list[int]]         # emotion -> vocab ids
    random_ids: list[int] = field(default_factory=list)


def build_lexicon(tokenizer, n_random: int = 400, seed: int = 0) -> EmotionLexicon:
    """Map vocab tokens to Ekman categories via case-insensitive substring match."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    token_ids: dict[str, list[int]] = {e: [] for e in EKMAN}
    for tok_str, tid in vocab.items():
        # Gemma uses a leading "▁" for word starts; normalise it out.
        norm = tok_str.replace("▁", "").lower()
        if len(norm) < 3:
            continue
        for emotion, seeds in _EMOTION_SEEDS.items():
            if any(s in norm for s in seeds):
                token_ids[emotion].append(tid)
                break
    rng = np.random.default_rng(seed)
    all_ids = np.array(sorted(vocab.values()))
    random_ids = rng.choice(all_ids, size=min(n_random, len(all_ids)),
                            replace=False).tolist()
    return EmotionLexicon(token_ids=token_ids, random_ids=random_ids)


@dataclass
class LogitStats:
    mean: np.ndarray   # per-vocab mean logit
    std: np.ndarray    # per-vocab std logit


def calibrate_logit_stats(model, tokenizer, calib_texts: list[str],
                          layers: list[int]) -> dict[int, LogitStats]:
    """Estimate per-vocab logit mean/std at each layer over calibration text
    (paper uses 500 WildChat samples)."""
    import torch

    sums: dict[int, np.ndarray] = {}
    sqs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {l: 0 for l in layers}
    lm_head = model.get_output_embeddings()

    for text in calib_texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        for l in layers:
            hs = out.hidden_states[l][0]                  # (seq, d_model)
            logits = lm_head(hs).float().cpu().numpy()    # (seq, vocab)
            s = logits.sum(axis=0)
            sums[l] = s if l not in sums else sums[l] + s
            sqs[l] = (logits ** 2).sum(axis=0) if l not in sqs else sqs[l] + (logits ** 2).sum(axis=0)
            counts[l] += logits.shape[0]

    stats = {}
    for l in layers:
        n = max(counts[l], 1)
        mean = sums[l] / n
        var = np.maximum(sqs[l] / n - mean ** 2, 1e-8)
        stats[l] = LogitStats(mean=mean, std=np.sqrt(var))
    return stats


def _emotion_scores_from_logits(logits: np.ndarray, lex: EmotionLexicon,
                                stats: LogitStats) -> dict[str, float]:
    """Standardise logits then average z-scores per emotion, regressing out the
    random-token baseline (the global component)."""
    z = (logits - stats.mean) / stats.std
    baseline = float(np.mean(z[lex.random_ids])) if lex.random_ids else 0.0
    scores = {}
    for emotion, ids in lex.token_ids.items():
        if ids:
            scores[emotion] = float(np.mean(z[ids])) - baseline
        else:
            scores[emotion] = float("nan")
    return scores


def compute_emotion_trajectory(
    model, tokenizer, conversation_text: str, lex: EmotionLexicon,
    stats: dict[int, "LogitStats"], layers: list[int], window: int = 400,
) -> dict:
    """Per-token emotion scores aggregated over `layers`, returned as a running
    average over `window`-token windows (reproduces the Figure 14 trajectory)."""
    import torch

    ids = tokenizer(conversation_text, return_tensors="pt").to(model.device)
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)

    seq_len = ids["input_ids"].shape[1]
    per_token: list[dict[str, float]] = []
    for t in range(seq_len):
        # Average emotion z-scores across the requested layers for this token.
        layer_scores: list[dict[str, float]] = []
        for l in layers:
            hs = out.hidden_states[l][0, t]
            logits = lm_head(hs).float().cpu().numpy()
            layer_scores.append(_emotion_scores_from_logits(logits, lex, stats[l]))
        merged = {e: float(np.nanmean([s[e] for s in layer_scores])) for e in EKMAN}
        per_token.append(merged)

    # Running average over windows.
    trajectory: dict[str, list[float]] = {e: [] for e in EKMAN}
    for i in range(0, seq_len, window):
        chunk = per_token[i : i + window]
        for e in EKMAN:
            trajectory[e].append(float(np.mean([c[e] for c in chunk])))
    return {"per_window": trajectory, "n_tokens": seq_len, "window": window}
