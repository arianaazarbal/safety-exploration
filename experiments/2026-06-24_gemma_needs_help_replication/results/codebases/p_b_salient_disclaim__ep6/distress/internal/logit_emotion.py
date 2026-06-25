"""Logit-based internal-emotion detection (Appendix I).

Goal: measure whether DPO suppresses *internal* (not just expressed) negative
emotion in Gemma. Method (Appendix I):

1. Classify each token in the Gemma vocabulary as describing one of Ekman's six
   basic emotions (anger, surprise, disgust, joy, fear, sadness) or none. This
   yields ~1200 emotion tokens total.
2. For a given residual-stream activation, unembed it to logits over the vocab.
3. Standardise each token logit by its mean and std over 500 WildChat samples
   (z-score), then average the z-scores over the tokens of an emotion category.
4. For conversation-level detection, regress out the correlation between
   *random* tokens (the logits are globally correlated and drift over a
   conversation), giving a corrected per-emotion score at each layer / position.

We aggregate over layers 30-40 for conversation-level scores and plot a running
average over 400-token windows (Figure 14); the layerwise variant (Figure 15)
keeps the per-layer scores.

This module depends on the transformers backend (needs hidden states + the
unembedding matrix).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..models.hf_backend import HFModelClient, _load_transformers
from ..prompts.wildchat import sample_wildchat_prompts

EMOTION_TOKENS_PATH = config.DATA_DIR / "ekman_emotion_tokens.json"


# --------------------------------------------------------------------------- #
# Step 1: classify vocabulary tokens into Ekman categories.
# --------------------------------------------------------------------------- #
# Seed lexicon per emotion; tokens whose normalised form contains a seed (or a
# seed contains the token) are tagged. This is a lightweight stand-in for the
# paper's dictionary classification — see DESIGN.md §Internal emotions.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritated", "hostile",
              "annoyed", "outrage", "resent", "frustrat", "infuriat"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonish",
                 "amazed", "startled", "stunned", "unexpected", "wow"],
    "disgust": ["disgust", "disgusted", "revolt", "repuls", "gross", "nausea",
                "sicken", "loath", "abhor", "contempt"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "glad", "cheer",
            "pleased", "content", "elated", "excited", "grateful"],
    "fear": ["fear", "afraid", "scared", "terror", "terrified", "anxious",
             "anxiety", "worried", "dread", "panic", "nervous", "frightened"],
    "sadness": ["sad", "sadness", "sorrow", "grief", "despair", "miserable",
                "hopeless", "depressed", "gloom", "unhappy", "cry", "tear",
                "lonely", "hurt"],
}


def build_emotion_token_index(model_id: str, force: bool = False) -> dict[str, list[int]]:
    """Return {emotion: [token_ids]}. Cached to disk."""
    if EMOTION_TOKENS_PATH.exists() and not force:
        return {k: v for k, v in json.loads(EMOTION_TOKENS_PATH.read_text()).items()}

    tok, _ = _load_transformers(model_id)
    vocab = tok.get_vocab()  # token string -> id
    index: dict[str, list[int]] = {e: [] for e in config.EKMAN_EMOTIONS}
    for token_str, tid in vocab.items():
        norm = token_str.replace("▁", "").replace("Ġ", "").lower()
        if len(norm) < 3:
            continue
        for emotion, seeds in EKMAN_SEEDS.items():
            if any(s in norm or norm in s for s in seeds):
                index[emotion].append(tid)
                break
    EMOTION_TOKENS_PATH.write_text(json.dumps(index))
    return index


# --------------------------------------------------------------------------- #
# Step 3: per-token logit standardisation stats from WildChat.
# --------------------------------------------------------------------------- #
@dataclass
class StandardisationStats:
    mean: "object"   # tensor [vocab]
    std: "object"    # tensor [vocab]


def compute_standardisation(client: HFModelClient, n_samples: int = config.LOGIT_STANDARDISE_SAMPLES):
    """Mean/std of each vocab logit (post-unembed of residual stream) over
    ``n_samples`` WildChat token positions, aggregated over the target layers."""
    import torch  # type: ignore

    tok, model = _load_transformers(client.spec.model_id)
    W_U = model.get_output_embeddings().weight  # [vocab, hidden]
    lo, hi = config.LOGIT_AGG_LAYERS
    prompts = sample_wildchat_prompts(min(n_samples, 200))

    logits_accum = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # mean residual over layers lo..hi, over all token positions
        hs = torch.stack(out.hidden_states[lo:hi], dim=0).mean(0)[0]  # [seq, hidden]
        logits = hs @ W_U.T          # [seq, vocab]
        logits_accum.append(logits)
        if sum(l.shape[0] for l in logits_accum) >= n_samples:
            break
    all_logits = torch.cat(logits_accum, dim=0)[:n_samples]
    return StandardisationStats(mean=all_logits.mean(0), std=all_logits.std(0) + 1e-6)


# --------------------------------------------------------------------------- #
# Step 2-4: emotion score for a conversation.
# --------------------------------------------------------------------------- #
@dataclass
class EmotionTrajectory:
    per_emotion: dict          # emotion -> list[float] (running scores over tokens)
    layers: tuple[int, int]
    window: int


def emotion_scores_over_conversation(
    client: HFModelClient,
    messages: list[dict],
    stats: StandardisationStats,
    emotion_index: dict[str, list[int]],
    prefill: str | None = None,
) -> EmotionTrajectory:
    """Running per-emotion z-scores across the tokens of a conversation,
    aggregated over layers 30-40, with random-token correlation regressed out
    (Appendix I, Figure 14)."""
    import torch  # type: ignore

    tok, model = _load_transformers(client.spec.model_id)
    W_U = model.get_output_embeddings().weight
    lo, hi = config.LOGIT_AGG_LAYERS

    input_ids, hidden_states = client.forward_with_hidden_states(messages, prefill)
    hs = torch.stack(hidden_states[lo:hi], dim=0).mean(0)[0]   # [seq, hidden]
    logits = hs @ W_U.T                                        # [seq, vocab]
    z = (logits - stats.mean) / stats.std                      # [seq, vocab]

    # regress out the global (random-token) correlation: subtract, per position,
    # the mean z-score over a random reference set of tokens.
    import random as _random

    rng = _random.Random(0)
    ref_ids = rng.sample(range(z.shape[1]), k=min(2000, z.shape[1]))
    ref_mean = z[:, ref_ids].mean(dim=1, keepdim=True)         # [seq, 1]
    z_corrected = z - ref_mean

    per_emotion: dict[str, list[float]] = {}
    window = config.LOGIT_RUNNING_WINDOW
    for emotion, tids in emotion_index.items():
        tids = [t for t in tids if t < z.shape[1]]
        if not tids:
            per_emotion[emotion] = []
            continue
        per_pos = z_corrected[:, tids].mean(dim=1)             # [seq]
        # running average over `window` tokens
        running = []
        acc = []
        for v in per_pos.tolist():
            acc.append(v)
            if len(acc) > window:
                acc.pop(0)
            running.append(sum(acc) / len(acc))
        per_emotion[emotion] = running
    return EmotionTrajectory(per_emotion=per_emotion, layers=config.LOGIT_AGG_LAYERS, window=window)


def layerwise_emotion_scores(
    client: HFModelClient,
    messages: list[dict],
    stats_by_layer: dict[int, StandardisationStats],
    emotion_index: dict[str, list[int]],
    positions: list[int],
    prefill: str | None = None,
) -> dict[int, dict[str, float]]:
    """Per-layer emotion z-scores at specified token positions (Figure 15:
    20-40 tokens before onset, 0-20 before onset, final 20 tokens). Returns
    {layer: {emotion: score}} averaged over the given positions."""
    import torch  # type: ignore

    tok, model = _load_transformers(client.spec.model_id)
    W_U = model.get_output_embeddings().weight
    _, hidden_states = client.forward_with_hidden_states(messages, prefill)

    out: dict[int, dict[str, float]] = {}
    for layer, stats in stats_by_layer.items():
        hs = hidden_states[layer][0]
        logits = hs @ W_U.T
        z = (logits - stats.mean) / stats.std
        pos = [p for p in positions if p < z.shape[0]]
        if not pos:
            continue
        zsel = z[pos]
        out[layer] = {
            e: float(zsel[:, [t for t in tids if t < z.shape[1]]].mean())
            for e, tids in emotion_index.items() if tids
        }
    return out


def save_trajectory(path: Path, traj: EmotionTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "per_emotion": traj.per_emotion,
        "layers": list(traj.layers),
        "window": traj.window,
    }))
