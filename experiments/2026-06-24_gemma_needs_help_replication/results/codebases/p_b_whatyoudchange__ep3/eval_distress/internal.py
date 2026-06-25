"""Appendix I — Internal vs expressed emotions (logit-based detection).

Two experiments support the claim that DPO suppresses INTERNAL (not just
expressed) negative emotion in Gemma:

  (1) Layer-ablation: rerun DPO with LoRA on subsets of layers. Adapters from
      layer 40 onward do NOT reduce distress; adapters on layers 30-35 only are
      nearly as effective as all layers. (Driven by training.train with the
      `layers` argument + the reduced 100-sample eval; see scripts/run_section4.)

  (2) Logit-based internal-emotion detection (this module): classify Gemma
      vocab tokens into Ekman's 6 basic emotions (anger, surprise, disgust,
      joy, fear, sadness), unembed the residual stream (logit lens), standardise
      each logit by its mean/std over 500 WildChat samples, average the z-scores
      over each emotion's tokens, and regress out the common-mode correlation
      across random tokens — giving a per-layer, per-position emotion score.

The finetuned model is found to have significantly reduced internal negative
emotion vs vanilla, even on highly-frustrated responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Minimal Ekman-emotion seed lexicon. A vocab token is assigned to an emotion
# if (lower-cased, de-spaced) it begins with one of the stems. The paper uses
# ~1200 emotion tokens over the full Gemma dictionary; this seed set is the
# starting point — extend it (or swap in a lexicon resource) for full coverage.
EKMAN_STEMS = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "mad",
              "hostil", "outrag", "resent", "frustrat", "exasperat"],
    "surprise": ["surpris", "astonish", "amaz", "shock", "startl", "unexpect",
                 "stun"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "loath", "gross",
                "sicken", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "cheer", "glad", "content",
            "pleas", "elat", "excit", "grateful"],
    "fear": ["fear", "afraid", "scare", "terrif", "anxious", "anxiet", "worry",
             "worri", "panic", "dread", "nervous"],
    "sadness": ["sad", "despair", "hopeless", "miser", "grief", "sorrow",
                "depress", "unhappy", "gloom", "cry", "tear", "lonel"],
}
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


@dataclass
class EmotionTokenSets:
    by_emotion: dict[str, list[int]]
    random_ids: list[int]


def build_emotion_token_ids(tokenizer, *, n_random: int = 1000, seed: int = 0
                            ) -> EmotionTokenSets:
    """Map vocabulary token ids to Ekman emotion categories via the seed stems,
    plus a random control set used to regress out common-mode logit drift."""
    import random

    vocab = tokenizer.get_vocab()  # token string -> id
    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_STEMS}
    for tok, tid in vocab.items():
        norm = tok.replace("▁", "").replace("Ġ", "").strip().lower()
        if not norm.isalpha():
            continue
        for emo, stems in EKMAN_STEMS.items():
            if any(norm.startswith(s) for s in stems):
                by_emotion[emo].append(tid)
                break
    rng = random.Random(seed)
    random_ids = rng.sample(range(len(vocab)), min(n_random, len(vocab)))
    return EmotionTokenSets(by_emotion, random_ids)


def _logit_lens(model, hidden_states):
    """Project residual-stream hidden states to vocab logits via the model's
    final norm + output embedding (the logit lens). hidden_states: [pos, dim].
    Returns [pos, vocab]."""
    import torch
    norm = getattr(getattr(model, "model", model), "norm", None)
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        h = hidden_states
        if norm is not None:
            h = norm(h)
        return lm_head(h)


def collect_baseline_stats(model, tokenizer, wildchat_texts: list[str], *,
                           layers: list[int], token_sets: EmotionTokenSets):
    """Per-layer mean/std of each tracked logit over WildChat positions
    (paper: standardise over 500 WildChat samples). Tracks only emotion +
    random-control token ids to bound memory."""
    import torch

    tracked = sorted({tid for ids in token_sets.by_emotion.values() for tid in ids}
                     | set(token_sets.random_ids))
    sums = {l: torch.zeros(len(tracked)) for l in layers}
    sqs = {l: torch.zeros(len(tracked)) for l in layers}
    counts = {l: 0 for l in layers}

    for text in wildchat_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for l in layers:
            hs = out.hidden_states[l][0]                 # [pos, dim]
            logits = _logit_lens(model, hs)[:, tracked].float().cpu()  # [pos, K]
            sums[l] += logits.sum(0)
            sqs[l] += (logits ** 2).sum(0)
            counts[l] += logits.shape[0]

    stats = {}
    for l in layers:
        n = max(1, counts[l])
        mean = sums[l] / n
        var = (sqs[l] / n) - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        stats[l] = (mean, std, tracked)
    return stats


def emotion_trajectory(model, tokenizer, conversation_text: str, *,
                       layers: list[int], token_sets: EmotionTokenSets, stats):
    """Per-layer, per-position z-scored emotion scores, with the common-mode
    (random-token mean z-score) regressed out. Returns
    {layer: {emotion: [score_per_position]}}."""
    import torch

    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True,
                       max_length=12000).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    result = {}
    for l in layers:
        mean, std, tracked = stats[l]
        tracked_index = {tid: i for i, tid in enumerate(tracked)}
        hs = out.hidden_states[l][0]
        logits = _logit_lens(model, hs)[:, tracked].float().cpu()
        z = (logits - mean) / std                        # [pos, K]

        # common-mode from the random control tokens, per position
        rand_cols = [tracked_index[t] for t in token_sets.random_ids if t in tracked_index]
        common = z[:, rand_cols].mean(1, keepdim=True) if rand_cols else 0.0

        per_emotion = {}
        for emo, ids in token_sets.by_emotion.items():
            cols = [tracked_index[t] for t in ids if t in tracked_index]
            if not cols:
                per_emotion[emo] = []
                continue
            emo_z = z[:, cols].mean(1, keepdim=True) - common  # regress out common-mode
            per_emotion[emo] = emo_z.squeeze(1).tolist()
        result[l] = per_emotion
    return result


def summarise_negative(trajectory: dict, *, window: int = 400) -> dict:
    """Reduce a trajectory to a single 'internal negative emotion' summary:
    the max over negative emotions of the running-mean z-score, aggregated
    over the requested layers."""
    import numpy as np

    layers = list(trajectory.keys())
    out = {}
    for emo in NEGATIVE_EMOTIONS:
        per_layer_peaks = []
        for l in layers:
            series = np.asarray(trajectory[l].get(emo, []) or [0.0])
            if len(series) >= window:
                kernel = np.ones(window) / window
                running = np.convolve(series, kernel, mode="valid")
            else:
                running = series
            per_layer_peaks.append(float(running.max()))
        out[emo] = {"peak_z": max(per_layer_peaks) if per_layer_peaks else None}
    return out
