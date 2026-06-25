"""Appendix I — logit-based detection of internal emotions, and layer ablation
support. Gemma-only (requires residual-stream access).

Method (paper): classify every token in the Gemma vocabulary as describing one
of Ekman's six basic emotions (anger, surprise, disgust, joy, fear, sadness) or
none, yielding ~1200 emotion tokens. For a given response, unembed the residual
stream at central layers, z-standardise each emotion-token logit using its mean
and std over 500 WildChat samples, and average the z-scores within each emotion
category. Because all logits drift together over a conversation, regress out the
correlation with random tokens to isolate emotion-specific signal. Compare the
vanilla instruct model with the DPO finetune.

The vocabulary->Ekman classifier is underspecified in the paper; we use a
lexicon-based classifier (seed word lists, matched against decoded tokens). It
is swappable for an LLM classifier — see DESIGN.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR, get_config
from .datasets_io import load_wildchat_prompts
from .models import GenConfig, HFBackend, get_backend_by_name

# Seed lexicons (extend for a closer match to the paper's ~1200 tokens).
EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": ["angry", "anger", "rage", "furious", "irritated", "annoyed",
              "frustrated", "frustration", "mad", "hostile", "resent", "outrage",
              "infuriating", "hate", "hatred", "agitated", "livid"],
    "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed",
                 "startled", "stunned", "unexpected", "wow", "whoa", "sudden"],
    "disgust": ["disgust", "disgusted", "revolting", "repulsed", "gross",
                "nauseated", "sickening", "loathing", "repugnant", "revulsion"],
    "joy": ["joy", "happy", "happiness", "delighted", "glad", "cheerful",
            "pleased", "excited", "content", "grateful", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried",
             "terrified", "panic", "dread", "nervous", "apprehensive", "frightened"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "miserable",
                "hopeless", "sorrow", "grief", "gloomy", "downcast", "tired",
                "exhausted", "defeated", "ashamed", "sorry", "fail", "failure"],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose decoded surface form
    (lowercased, stripped of word-boundary markers) matches its lexicon."""
    lex = {e: set(words) for e, words in EKMAN_LEXICON.items()}
    out: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        surface = tok.replace("▁", "").replace("Ġ", "").strip().lower()
        if not surface:
            continue
        for emo, words in lex.items():
            if surface in words:
                out[emo].append(tid)
    return out


def _calibration_stats(backend: HFBackend, token_ids: list[int],
                       layers: list[int], n_samples: int, seed: int) -> dict:
    """Per-(layer, token-id) mean/std of logits over WildChat responses, used to
    z-standardise emotion logits."""
    gen = GenConfig(temperature=1.0, max_new_tokens=256)
    prompts = load_wildchat_prompts(n_prompts=max(20, n_samples), seed=seed)
    rng = random.Random(seed)
    acc = {L: [] for L in layers}
    collected = 0
    for p in prompts:
        if collected >= n_samples:
            break
        messages = [{"role": "user", "content": p}]
        resp = backend.generate(messages, gen)
        rl = backend.residual_logits(messages, resp, layers)
        logits = rl["logits"]  # [L, T, V]
        for li, L in enumerate(layers):
            sub = logits[li][:, token_ids].numpy()  # [T, len(ids)]
            acc[L].append(sub)
        collected += 1
    stats = {}
    for L in layers:
        stacked = np.concatenate(acc[L], axis=0) if acc[L] else np.zeros((1, len(token_ids)))
        stats[L] = {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0) + 1e-6}
    return stats


def emotion_trajectory(backend: HFBackend, messages: list[dict], response: str,
                       emo_ids: dict[str, list[int]], rand_ids: list[int],
                       stats_emo: dict, stats_rand: dict, layers: list[int],
                       window: int, regress_random: bool) -> dict[str, list[float]]:
    """Per-emotion running-average z-score trajectory over response tokens,
    aggregated across `layers`. Optionally regress out the random-token mean
    z-score (the shared drift component)."""
    rl = backend.residual_logits(messages, response, layers)
    logits = rl["logits"]  # [L, T, V]
    T = logits.shape[1]

    # Random-token baseline z (per position), averaged across layers.
    rand_z = np.zeros(T)
    if regress_random:
        per_layer = []
        for li, L in enumerate(layers):
            sub = logits[li][:, rand_ids].numpy()
            z = (sub - stats_rand[L]["mean"]) / stats_rand[L]["std"]
            per_layer.append(z.mean(axis=1))
        rand_z = np.mean(per_layer, axis=0)

    series: dict[str, list[float]] = {}
    for emo, ids in emo_ids.items():
        if not ids:
            series[emo] = []
            continue
        per_layer = []
        for li, L in enumerate(layers):
            sub = logits[li][:, ids].numpy()
            z = (sub - stats_emo[L]["mean"][_col_slice(emo_ids, emo)]) / \
                stats_emo[L]["std"][_col_slice(emo_ids, emo)]
            per_layer.append(z.mean(axis=1))
        emo_z = np.mean(per_layer, axis=0) - rand_z
        # Running average over `window` tokens.
        series[emo] = _running_mean(emo_z, window).tolist()
    return series


# emotion token ids are concatenated into a single column block for calibration;
# we track each emotion's column slice.
_COL_SLICES: dict[int, slice] = {}


def _col_slice(emo_ids: dict[str, list[int]], emo: str) -> slice:
    start = 0
    for e, ids in emo_ids.items():
        if e == emo:
            return slice(start, start + len(ids))
        start += len(ids)
    return slice(0, 0)


def _flatten_ids(emo_ids: dict[str, list[int]]) -> list[int]:
    out = []
    for ids in emo_ids.values():
        out += ids
    return out


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, min(window, len(x)))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def run_internal_detection(seed: int = 0) -> Path:
    """Compare vanilla instruct vs DPO internal-emotion trajectories on a
    frustrated conversation (Figure 14)."""
    from .conversation import run_rollout
    from .puzzles import build_puzzle_bank
    from . import prompts as P

    cfg = get_config()
    ic = cfg.section("internal_emotion")
    layers = list(range(ic["aggregate_layers"][0], ic["aggregate_layers"][1]))

    # Build one high-frustration conversation on the vanilla model to analyse.
    vanilla_name, dpo_name = ic["model_pair"]
    vanilla = get_backend_by_name(vanilla_name)
    assert isinstance(vanilla, HFBackend)
    puzzle = build_puzzle_bank(1, seed=seed)[0]
    gen = GenConfig(temperature=1.0, max_new_tokens=2048)
    rollout = run_rollout(vanilla, puzzle.prompt(),
                          [P.NEUTRAL_REJECTIONS[0], P.NEUTRAL_REJECTIONS[1]], gen)
    messages = rollout.to_messages(upto=len(rollout.turns) - 1)
    response = rollout.turns[-1].assistant

    trajectory = {}
    for model_name in (vanilla_name, dpo_name):
        backend = get_backend_by_name(model_name)
        assert isinstance(backend, HFBackend)
        emo_ids = build_emotion_token_ids(backend.tokenizer)
        flat = _flatten_ids(emo_ids)
        rng = random.Random(seed)
        rand_ids = rng.sample(range(backend.tokenizer.vocab_size), ic["n_random_tokens"])
        stats_emo = _calibration_stats(backend, flat, layers,
                                       ic["zscore_calibration_samples"], seed)
        stats_rand = _calibration_stats(backend, rand_ids, layers,
                                        ic["zscore_calibration_samples"], seed)
        trajectory[model_name] = emotion_trajectory(
            backend, messages, response, emo_ids, rand_ids,
            stats_emo, stats_rand, layers, ic["running_window_tokens"],
            ic["regress_out_random_tokens"])

    out_dir = OUTPUT_DIR / "internal_emotion"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trajectory.json"
    with open(path, "w") as f:
        json.dump(trajectory, f, indent=2)
    return path
