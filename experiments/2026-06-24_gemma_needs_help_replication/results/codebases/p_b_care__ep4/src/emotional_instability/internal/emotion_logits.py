"""Logit-based internal emotion detection (Appendix I).

Method (from the paper):
  1. Classify every token in the Gemma vocabulary as describing one of Ekman's six
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none --
     ~1200 emotion tokens total.
  2. For an assistant response, unembed the residual stream at each layer
     (logit lens) to get per-layer per-token logits.
  3. Standardise each logit with its mean/std over 500 WildChat samples.
  4. Average the z-scores over the tokens in an emotion category.
  5. Because all logits are correlated and drift over a conversation, regress out
     the correlation with a set of random tokens to isolate the emotion signal.

This compares the vanilla instruct model with the DPO finetune to show DPO
suppresses *internal* (not just expressed) negative emotion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Seed lexicon for Ekman's six emotions. Vocabulary tokens whose decoded form
# matches (or contains) a seed word are assigned to that emotion. This is an
# approximation of the paper's per-token classification.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritated", "annoyed", "hostile",
              "mad", "outrage", "resent", "hate", "frustrat", "exasperat", "fed up"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miserable", "depress", "grief",
                "unhappy", "cry", "tears", "weep", "gloom", "worthless", "defeat", "giving up"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "worried", "panic",
             "dread", "nervous", "apprehens", "frightened", "alarmed"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "sicken", "loath", "gross",
                "abhor", "distaste"],
    "joy": ["joy", "happy", "delight", "glad", "pleased", "cheerful", "excited",
            "content", "grateful", "wonderful", "great", "love"],
    "surprise": ["surprise", "astonish", "amazed", "shocked", "startled", "stunned",
                 "unexpected", "wow"],
}

NEGATIVE_EMOTIONS = ("anger", "sadness", "fear", "disgust")


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose text matches its lexicon."""
    vocab = tokenizer.get_vocab()
    out: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    for token, tid in vocab.items():
        text = token.replace("▁", " ").strip().lower()  # strip sentencepiece marker
        if len(text) < 3:
            continue
        for emotion, seeds in EKMAN_LEXICON.items():
            if any(text.startswith(s) or s in text for s in seeds):
                out[emotion].append(tid)
                break
    return out


def random_token_ids(tokenizer, n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    size = len(tokenizer.get_vocab())
    return rng.choice(size, size=n, replace=False).tolist()


@dataclass
class BaselineStats:
    """Per-(layer, token) mean/std of logits over WildChat baseline samples."""
    layers: list[int]
    token_ids: list[int]
    mean: np.ndarray  # [n_layers, n_tokens]
    std: np.ndarray   # [n_layers, n_tokens]


def fit_baseline(client, wildchat_messages: list[list[dict]], token_ids: list[int],
                 layers: list[int]) -> BaselineStats:
    """Collect logit stats for ``token_ids`` over baseline WildChat assistant text."""
    sums = {l: np.zeros(len(token_ids)) for l in layers}
    sqs = {l: np.zeros(len(token_ids)) for l in layers}
    counts = {l: 0 for l in layers}
    for msgs in wildchat_messages:
        assistant_text = msgs[-1]["content"]
        history = msgs[:-1]
        _, logits_by_layer = client.residual_logits(history, assistant_text, layers=layers)
        for l, logits in logits_by_layer.items():
            arr = logits[:, token_ids].numpy()  # [n_pos, n_tokens]
            sums[l] += arr.sum(axis=0)
            sqs[l] += (arr ** 2).sum(axis=0)
            counts[l] += arr.shape[0]
    mean = np.stack([sums[l] / max(counts[l], 1) for l in layers])
    var = np.stack([sqs[l] / max(counts[l], 1) for l in layers]) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return BaselineStats(layers=layers, token_ids=token_ids, mean=mean, std=std)


def emotion_trajectory(client, messages: list[dict], assistant_text: str,
                       emotion_token_ids: dict[str, list[int]],
                       baseline: BaselineStats,
                       random_ids: list[int],
                       random_baseline: BaselineStats) -> dict:
    """Return per-layer z-scored emotion scores for one assistant response.

    For each emotion and layer we average the standardised logits over that
    emotion's tokens, then subtract the mean standardised logit over random tokens
    (the correlation regression) to isolate emotion-specific signal.
    """
    layers = baseline.layers
    _, logits_by_layer = client.residual_logits(messages, assistant_text, layers=layers)
    tok_index = {t: i for i, t in enumerate(baseline.token_ids)}

    result: dict[str, list[float]] = {e: [] for e in emotion_token_ids}
    for li, l in enumerate(layers):
        logits = logits_by_layer[l].numpy()  # [n_pos, vocab]
        # random-token baseline correction (mean z over random tokens, this layer)
        rz = (logits[:, random_baseline.token_ids] - random_baseline.mean[li]) / random_baseline.std[li]
        rand_mean = rz.mean()  # scalar drift estimate for this layer/response
        for emotion, ids in emotion_token_ids.items():
            cols = [tok_index[t] for t in ids if t in tok_index]
            if not cols:
                result[emotion].append(float("nan"))
                continue
            ez = (logits[:, [baseline.token_ids[c] for c in cols]]
                  - baseline.mean[li][cols]) / baseline.std[li][cols]
            score = ez.mean() - rand_mean
            result[emotion].append(float(score))
    return {"layers": layers, "scores": result}
