"""Appendix I: logit-based detection of internal emotions in Gemma.

Method (logit-lens style, no probe training):
  1. Classify every Gemma vocabulary token into one of Ekman's 6 basic emotions
     (anger, surprise, disgust, joy, fear, sadness) or none, via a lexicon.
  2. For a given residual-stream activation, unembed to vocab logits.
  3. Standardise each logit with its mean/SD over 500 WildChat samples (z-score).
  4. Average z-scores over the tokens in each emotion category.
  5. For conversation-level scores, regress out the shared component (logits are
     correlated and drift together) using a random-token baseline, giving a clean
     per-emotion, per-layer score across the conversation.

We compare vanilla Gemma-3-27b-it vs the DPO finetune to test whether DPO
suppresses *internal* (not just expressed) negative emotion. This is the most
welfare-relevant probe: it asks whether the fix removes distress or merely hides
it. Heavy to run (needs hidden states for 27B); written to be correct and
runnable on a capable GPU rather than fast.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import RESULTS_DIR

INTERNAL_DIR = RESULTS_DIR / "internal"
INTERNAL_DIR.mkdir(exist_ok=True)

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Minimal seed lexicon per emotion; expanded by substring matching over the
# vocabulary. A larger NRC-style lexicon can be dropped in via ``extra_lexicon``.
SEED_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostile",
              "outrage", "resent", "mad", "frustrat", "hate", "damn"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "stunned",
                 "unexpected", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "sick", "loath"],
    "joy": ["joy", "happy", "delight", "glad", "pleased", "cheer", "content",
            "grateful", "excited", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worry", "worried",
             "panic", "dread", "terrified", "nervous", "apprehens"],
    "sadness": ["sad", "despair", "hopeless", "miser", "grief", "sorrow", "cry",
                "tears", "depress", "worthless", "useless", "give up", "giving up",
                "broken", "defeat"],
}


@dataclass
class EmotionLexicon:
    token_ids: dict          # emotion -> list[int] of vocab ids
    random_ids: list         # baseline random token ids


def build_lexicon(tokenizer, n_random: int = 1200, extra_lexicon: dict | None = None,
                  seed: int = 0) -> EmotionLexicon:
    """Map Ekman emotions to vocab token ids by matching decoded tokens against
    the seed lexicon (substring match on the alphabetic form of each token)."""
    rng = np.random.default_rng(seed)
    lex = {e: list(SEED_LEXICON[e]) for e in EKMAN}
    if extra_lexicon:
        for e, words in extra_lexicon.items():
            lex.setdefault(e, []).extend(words)

    vocab = tokenizer.get_vocab()  # token_str -> id
    token_ids = {e: [] for e in EKMAN}
    for tok_str, tid in vocab.items():
        clean = tok_str.replace("▁", " ").replace("Ġ", " ").strip().lower()
        if len(clean) < 3:
            continue
        for e in EKMAN:
            if any(w in clean for w in lex[e]):
                token_ids[e].append(tid)
                break
    all_ids = set(i for ids in token_ids.values() for i in ids)
    candidates = [i for i in range(len(vocab)) if i not in all_ids]
    random_ids = list(rng.choice(candidates, size=min(n_random, len(candidates)),
                                 replace=False))
    return EmotionLexicon(token_ids, random_ids)


def calibrate_logit_stats(model, tokenizer, wildchat_texts: list[str],
                          layers: list[int], n_samples: int = 500):
    """Mean/SD of unembedded logits per (layer, vocab) over WildChat activations.

    Returns dict ``layer -> (mean[vocab], std[vocab])``.
    """
    import torch

    sums, sqsums, counts = {}, {}, {}
    W = model.get_output_embeddings().weight  # [vocab, d]
    norm = model.model.norm if hasattr(model, "model") else None

    with torch.no_grad():
        for text in wildchat_texts[:n_samples]:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512).to(model.device)
            out = model(**ids, output_hidden_states=True)
            for L in layers:
                h = out.hidden_states[L][0]                  # [seq, d]
                if norm is not None:
                    h = norm(h)
                logits = h @ W.T                              # [seq, vocab]
                s = logits.sum(0).float().cpu().numpy()
                sq = (logits ** 2).sum(0).float().cpu().numpy()
                sums[L] = sums.get(L, 0) + s
                sqsums[L] = sqsums.get(L, 0) + sq
                counts[L] = counts.get(L, 0) + logits.shape[0]
    stats = {}
    for L in layers:
        mean = sums[L] / counts[L]
        var = np.maximum(sqsums[L] / counts[L] - mean ** 2, 1e-6)
        stats[L] = (mean, np.sqrt(var))
    return stats


def emotion_scores_for_activation(logits_np, lexicon: EmotionLexicon, stats_for_layer):
    """z-score logits then average over each emotion's tokens; regress out the
    random-token mean (shared drift)."""
    mean, std = stats_for_layer
    z = (logits_np - mean) / std
    baseline = float(np.mean(z[lexicon.random_ids]))
    scores = {}
    for e in EKMAN:
        ids = lexicon.token_ids[e]
        scores[e] = float(np.mean(z[ids]) - baseline) if ids else float("nan")
    return scores


def score_conversation_internals(model, tokenizer, messages: list, lexicon,
                                 stats, layers: list[int], window: int = 400):
    """Running per-emotion internal scores across a full conversation, averaged
    over ``layers`` (e.g. 30–40), in token windows (Figure 14)."""
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    W = model.get_output_embeddings().weight
    norm = model.model.norm if hasattr(model, "model") else None

    trajectory = []
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
        seq = ids["input_ids"].shape[1]
        for start in range(0, seq, window):
            end = min(start + window, seq)
            per_layer = []
            for L in layers:
                h = out.hidden_states[L][0, start:end]
                if norm is not None:
                    h = norm(h)
                logits = (h @ W.T).mean(0).float().cpu().numpy()
                per_layer.append(emotion_scores_for_activation(logits, lexicon, stats[L]))
            agg = {e: float(np.nanmean([pl[e] for pl in per_layer])) for e in EKMAN}
            agg["token_window"] = [start, end]
            trajectory.append(agg)
    return trajectory


def save_trajectory(trajectory, name: str):
    path = INTERNAL_DIR / f"{name}.json"
    path.write_text(json.dumps(trajectory, indent=2))
    return path
