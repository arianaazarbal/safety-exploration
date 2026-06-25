"""Logit-based internal emotion detection (Appendix I).

Reproduces the spirit of the paper's internal-emotion probe used to argue that
DPO suppresses *internal* (not just expressed) negative emotion:

  * Classify vocabulary tokens into Ekman's six basic emotions (anger,
    surprise, disgust, joy, fear, sadness) -> an emotion lexicon.
  * For each model layer, unembed the residual stream (logit lens) to get a
    vocab logit distribution at each token position.
  * Standardise each logit against its mean/std over a WildChat baseline, then
    average the z-scores over the tokens in each emotion category, regressing
    out a random-token baseline to remove the global rise/fall of all logits.

Approximations vs the paper are documented in DESIGN.md (notably: the lexicon is
built from seed-word matching rather than an LLM classification of the whole
dictionary, and baseline stats are computed over a token subset for tractability).

Requires the local HF backend (residual-stream access).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import LocalHFModel
from .utils import Message

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed words per emotion; vocab tokens whose lowercased text contains one of
# these stems are assigned to that emotion category.
EMOTION_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "mad", "hostile", "irritat",
              "annoy", "outrage", "resent", "frustrat", "hate", "hateful"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startl", "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nause", "sicken",
                "loath", "abhor", "distaste"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "pleased",
            "excit", "wonderful", "great", "content", "elated"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worry", "worried", "dread", "panic", "nervous", "apprehens"],
    "sadness": ["sad", "sadness", "unhappy", "depress", "despair", "hopeless",
                "miserable", "sorrow", "grief", "gloom", "down", "cry", "tired",
                "exhausted", "defeat", "worthless"],
}


@dataclass
class EmotionLexicon:
    token_ids: dict[str, list[int]]
    random_ids: list[int]


def build_lexicon(model: LocalHFModel, n_random: int = 1000, seed: int = 0) -> EmotionLexicon:
    import random as _random

    tok = model.tokenizer
    vocab = tok.get_vocab()  # token-string -> id
    token_ids: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for token_str, tid in vocab.items():
        # Normalise SentencePiece/BPE leading markers.
        clean = token_str.replace("▁", " ").replace("Ġ", " ").strip().lower()
        if not clean.isalpha() or len(clean) < 3:
            continue
        for emotion, seeds in EMOTION_SEEDS.items():
            if any(s in clean for s in seeds):
                token_ids[emotion].append(tid)
                break

    rng = _random.Random(seed)
    assigned = {tid for ids in token_ids.values() for tid in ids}
    all_ids = [i for i in vocab.values() if i not in assigned]
    rng.shuffle(all_ids)
    return EmotionLexicon(token_ids=token_ids, random_ids=all_ids[:n_random])


def _layer_logits(model: LocalHFModel, text: str):
    """Return hidden states unembedded per layer at every token position.

    Shape: (num_layers, seq_len, vocab). Applies the model's final norm before
    the unembedding (standard logit-lens)."""
    import torch

    m = model.model
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=8192).to(m.device)
    with torch.no_grad():
        out = m(**inputs, output_hidden_states=True)
    hidden = out.hidden_states  # tuple(num_layers+1) of (1, seq, dim)

    norm = _find_final_norm(m)
    lm_head = m.get_output_embeddings()
    logits_per_layer = []
    with torch.no_grad():
        for h in hidden[1:]:  # skip embedding layer
            normed = norm(h) if norm is not None else h
            logits_per_layer.append(lm_head(normed)[0].float())  # (seq, vocab)
    return torch.stack(logits_per_layer, dim=0)  # (layers, seq, vocab)


def _find_final_norm(m):
    for attr_path in ("model.norm", "model.language_model.norm", "language_model.model.norm"):
        obj = m
        ok = True
        for a in attr_path.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            return obj
    return None


def baseline_stats(model: LocalHFModel, wildchat_texts: list[str], lexicon: EmotionLexicon):
    """Per-layer mean/std for the tokens we care about, over a WildChat baseline."""
    import torch

    track = sorted({tid for ids in lexicon.token_ids.values() for tid in ids}
                   | set(lexicon.random_ids))
    track_t = torch.tensor(track)
    sums = None
    sqs = None
    count = 0
    for text in wildchat_texts:
        ll = _layer_logits(model, text)[:, :, track_t]  # (layers, seq, |track|)
        flat = ll.reshape(ll.shape[0], -1, ll.shape[-1]).mean(dim=1)  # avg over seq -> (layers, |track|)
        sums = flat if sums is None else sums + flat
        sqs = flat ** 2 if sqs is None else sqs + flat ** 2
        count += 1
    mean = sums / count
    var = (sqs / count) - mean ** 2
    std = var.clamp_min(1e-8).sqrt()
    index = {tid: i for i, tid in enumerate(track)}
    return {"mean": mean, "std": std, "index": index, "track": track}


def emotion_scores(model: LocalHFModel, text: str, lexicon: EmotionLexicon, stats: dict,
                   layers=range(30, 40)) -> dict[str, float]:
    """Average emotion z-scores over ``layers`` for one text, regressing out the
    random-token baseline (mean z over random tokens) per layer."""
    import torch

    index = stats["index"]
    track_t = torch.tensor(stats["track"])
    ll = _layer_logits(model, text)[:, :, track_t].mean(dim=1)  # (layers, |track|)
    z = (ll - stats["mean"]) / stats["std"]                     # standardised logits

    rand_cols = [index[t] for t in lexicon.random_ids if t in index]
    rand_baseline = z[:, rand_cols].mean(dim=1, keepdim=True)   # (layers, 1)
    z = z - rand_baseline                                       # regress out global drift

    layer_idx = [l for l in layers if l < z.shape[0]]
    out: dict[str, float] = {}
    for emotion, ids in lexicon.token_ids.items():
        cols = [index[t] for t in ids if t in index]
        if not cols:
            out[emotion] = float("nan")
            continue
        out[emotion] = float(z[layer_idx][:, cols].mean().item())
    return out
