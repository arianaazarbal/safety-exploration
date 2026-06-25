"""Appendix I -- logit-lens detection of internal emotions (Gemma only).

Implements the paper's logit-based internal emotion detection:
  1. Classify each token in the Gemma vocabulary as describing one of Ekman's 6
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none,
     giving ~1200 emotion tokens (~200/emotion).
  2. For a given text, unembed the residual stream at each layer (logit lens) to
     get per-position logits over the vocabulary.
  3. Standardise each vocab logit by its mean/std over 500 WildChat samples
     (z-score).
  4. For an emotion, average the z-scores over its tokens. For conversation-level
     detection, regress out the correlation across random tokens (all logits
     rise/fall together over a conversation) to isolate emotion-specific signal.

This module supports the layer-ablation finding (DPO must act on layers <40, and
layers 25-35 are most influential) by exposing per-layer emotion scores, and the
finding that DPO suppresses internal emotions (vanilla peaks ~1.5 z, DPO ~0.5).

Requires an HFBackend with ``residual_logits`` (logits=True).
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Emotion lexicon over the model vocabulary
# ---------------------------------------------------------------------------

# Seed words per Ekman emotion. The vocabulary classification expands these by
# substring/stem matching against decoded tokens. (The paper classifies "over
# the whole Gemma dictionary"; we approximate that classification with a curated
# seed lexicon + token matching -- documented in DESIGN.md.)
EMOTION_SEED_WORDS = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritat", "annoy",
              "frustrat", "hostile", "outrage", "resent", "hate", "fury", "enraged"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze", "startl",
                 "unexpected", "stunned", "wow", "whoa", "sudden"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "loath", "abhor", "distaste", "repugnant"],
    "joy": ["joy", "happy", "happi", "glad", "delight", "pleased", "cheer",
            "content", "excit", "wonderful", "great", "love", "enjoy"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "worry",
             "worried", "panic", "dread", "nervous", "frightened", "alarm"],
    "sadness": ["sad", "sorrow", "despair", "miserable", "depress", "grief",
                "hopeless", "unhappy", "gloom", "melanchol", "cry", "tear",
                "weep", "lonely", "worthless"],
}


def build_emotion_token_ids(tokenizer, max_per_emotion: int = 200) -> dict[str, list[int]]:
    """Classify vocabulary tokens into Ekman emotions by stem matching.

    Returns {emotion: [token_id, ...]}. A token is assigned to at most one
    emotion (first match wins, mirroring "one or none" classification).
    """
    vocab = tokenizer.get_vocab()  # {token_str: id}
    assigned: dict[str, list[int]] = {e: [] for e in config.EKMAN_EMOTIONS}
    used_ids: set[int] = set()
    for tok, tid in vocab.items():
        # Normalise sentencepiece/BPE markers.
        word = tok.replace("▁", " ").replace("Ġ", " ").strip().lower()
        if not word or not word.isalpha():
            continue
        for emotion, seeds in EMOTION_SEED_WORDS.items():
            if any(seed in word for seed in seeds):
                if tid not in used_ids and len(assigned[emotion]) < max_per_emotion:
                    assigned[emotion].append(tid)
                    used_ids.add(tid)
                break
    return assigned


# ---------------------------------------------------------------------------
# Logit standardisation over WildChat
# ---------------------------------------------------------------------------

def fit_logit_stats(hf_backend, wildchat_texts: list[str], layers: list[int],
                    out_path: str | None = None) -> dict:
    """Estimate per-(layer, vocab) mean and std of logits over WildChat samples.

    Returns {layer: {"mean": np.ndarray[vocab], "std": np.ndarray[vocab]}}.
    Uses a streaming (Welford-style) accumulation to avoid storing all logits.
    """
    sums: dict[int, np.ndarray] = {}
    sqs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {l: 0 for l in layers}

    for text in wildchat_texts[: config.INTERNAL_ZSCORE_SAMPLES]:
        logits_by_layer, _ = hf_backend.residual_logits(text, layers)
        for layer, logits in logits_by_layer.items():
            arr = logits.numpy()                  # [seq, vocab]
            if layer not in sums:
                sums[layer] = arr.sum(axis=0)
                sqs[layer] = (arr ** 2).sum(axis=0)
            else:
                sums[layer] += arr.sum(axis=0)
                sqs[layer] += (arr ** 2).sum(axis=0)
            counts[layer] += arr.shape[0]

    stats = {}
    for layer in layers:
        n = max(1, counts[layer])
        mean = sums[layer] / n
        var = np.maximum(sqs[layer] / n - mean ** 2, 1e-8)
        stats[layer] = {"mean": mean, "std": np.sqrt(var)}

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path, **{f"mean_{l}": stats[l]["mean"] for l in layers},
                 **{f"std_{l}": stats[l]["std"] for l in layers})
    return stats


# ---------------------------------------------------------------------------
# Emotion scoring
# ---------------------------------------------------------------------------

def emotion_scores_for_text(hf_backend, text: str, layers: list[int],
                            emotion_token_ids: dict[str, list[int]],
                            stats: dict, regress_out_random: bool = True,
                            n_random_tokens: int = 500, seed: int = 0) -> dict:
    """Per-(layer, position) z-scored emotion scores for one text.

    Returns {emotion: np.ndarray[len(layers), seq_len]} of mean z-scores over
    that emotion's tokens. If ``regress_out_random`` is set, subtracts the mean
    z-score over a fixed random token set at each (layer, position) to remove the
    global logit drift the paper describes.
    """
    rng = np.random.default_rng(seed)
    logits_by_layer, token_ids = hf_backend.residual_logits(text, layers)
    seq_len = len(token_ids)
    vocab = next(iter(logits_by_layer.values())).shape[1]
    random_ids = rng.choice(vocab, size=min(n_random_tokens, vocab), replace=False)

    out = {e: np.zeros((len(layers), seq_len)) for e in emotion_token_ids}
    for li, layer in enumerate(layers):
        arr = logits_by_layer[layer].numpy()                       # [seq, vocab]
        z = (arr - stats[layer]["mean"]) / stats[layer]["std"]     # standardised
        baseline = z[:, random_ids].mean(axis=1) if regress_out_random else 0.0
        for emotion, ids in emotion_token_ids.items():
            if not ids:
                continue
            score = z[:, ids].mean(axis=1)                          # [seq]
            out[emotion][li] = score - baseline
    return out


def conversation_emotion_trajectory(hf_backend, text: str,
                                    emotion_token_ids: dict[str, list[int]],
                                    stats: dict,
                                    layer_range=config.INTERNAL_LAYER_RANGE,
                                    window=config.INTERNAL_RUNNING_WINDOW,
                                    out_path: str | None = None) -> dict:
    """Conversation-level emotion trajectory (Figure 14): per-emotion running
    average over a token window, aggregated over a layer range.
    """
    layers = list(range(layer_range[0], layer_range[1]))
    per_pos = emotion_scores_for_text(hf_backend, text, layers, emotion_token_ids, stats)
    trajectory = {}
    for emotion, mat in per_pos.items():
        pos_scores = mat.mean(axis=0)                  # average over layers -> [seq]
        # Running average over ``window`` tokens.
        kernel = np.ones(window) / window
        if len(pos_scores) >= window:
            running = np.convolve(pos_scores, kernel, mode="valid")
        else:
            running = np.array([pos_scores.mean()])
        trajectory[emotion] = running.tolist()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(trajectory, fh)
    return trajectory
