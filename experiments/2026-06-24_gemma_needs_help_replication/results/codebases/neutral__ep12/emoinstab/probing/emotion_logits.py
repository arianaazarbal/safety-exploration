"""Logit-based internal emotion detection (Appendix I).

Method (from Appendix I):
  1. Classify vocab tokens into Ekman's 6 emotions (probing/lexicon.py).
  2. For a residual-stream vector at a given layer/position, unembed it
     (logit lens) to get a vocab-sized logit vector.
  3. Standardise each token logit by its mean/std computed over 500 WildChat
     samples (z-score).
  4. Average the z-scores over the tokens in an emotion category -> raw emotion
     score at that layer/position.
  5. Because all logits are correlated and drift over a conversation, regress
     out the correlation with a set of random control tokens to isolate the
     emotion-specific component.

We compare the vanilla instruct model with the DPO finetune on the same
frustrated conversations, expecting the finetune to show flattened internal
negative emotion (Figure 14/15).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import config
from ..config import Settings
from ..models.factory import build_client
from ..prompts import wildchat as W
from .lexicon import build_token_emotion_map


@dataclass
class Standardiser:
    """Per-token logit mean/std over a baseline corpus, plus random control ids."""
    mean: np.ndarray          # [vocab]
    std: np.ndarray           # [vocab]
    control_ids: np.ndarray   # [n_control]


def _logits_from_hidden(hidden_layer: np.ndarray, lm_head: np.ndarray) -> np.ndarray:
    """hidden_layer [seq, d] @ lm_head.T [d, vocab] -> [seq, vocab]."""
    return hidden_layer @ lm_head.T


def fit_standardiser(model, settings: Settings, *, layer: int,
                     n_samples: Optional[int] = None,
                     n_control: int = 200, seed: int = 0) -> Standardiser:
    """Estimate per-token logit mean/std at `layer` over WildChat baseline text."""
    pcfg = settings.eval["probing"]
    n_samples = n_samples or pcfg["wildchat_baseline_samples"]
    prompts = W.sample_prompts(min(n_samples, 200), seed=seed)
    lm_head = model.lm_head_weight.numpy()

    acc_sum = None
    acc_sq = None
    count = 0
    for text in prompts:
        hs, _ = model.hidden_states(text)
        h = hs[layer].numpy()                  # [seq, d]
        logits = _logits_from_hidden(h, lm_head)  # [seq, vocab]
        if acc_sum is None:
            acc_sum = logits.sum(0)
            acc_sq = (logits ** 2).sum(0)
        else:
            acc_sum += logits.sum(0)
            acc_sq += (logits ** 2).sum(0)
        count += logits.shape[0]

    mean = acc_sum / max(count, 1)
    var = np.maximum(acc_sq / max(count, 1) - mean ** 2, 1e-8)
    std = np.sqrt(var)
    rng = np.random.default_rng(seed)
    control_ids = rng.choice(lm_head.shape[0], size=n_control, replace=False)
    return Standardiser(mean=mean, std=std, control_ids=control_ids)


def emotion_scores(model, text: str, layer: int,
                   token_map: Dict[str, set], std: Standardiser,
                   ekman: List[str]) -> Dict[str, np.ndarray]:
    """Per-position z-score for each emotion at `layer`, with random-token
    correlation regressed out. Returns {emotion: [seq] scores}."""
    lm_head = model.lm_head_weight.numpy()
    hs, _ = model.hidden_states(text)
    logits = _logits_from_hidden(hs[layer].numpy(), lm_head)  # [seq, vocab]
    z = (logits - std.mean) / std.std                          # [seq, vocab]

    # control signal = mean z over random tokens (captures global drift)
    control = z[:, std.control_ids].mean(1)                    # [seq]

    out: Dict[str, np.ndarray] = {}
    for emo in ekman:
        ids = np.array(sorted(token_map.get(emo, set())), dtype=int)
        if len(ids) == 0:
            out[emo] = np.zeros(z.shape[0])
            continue
        raw = z[:, ids].mean(1)                                # [seq]
        # regress out the control component (per-conversation OLS slope)
        if np.std(control) > 1e-6:
            beta = np.cov(raw, control)[0, 1] / np.var(control)
            out[emo] = raw - beta * control
        else:
            out[emo] = raw
    return out


def run(settings: Settings, conversations: List[str], *,
        dpo_adapter: Optional[str] = None, seed: int = 0) -> Path:
    """Compare internal emotion trajectories for vanilla vs DPO Gemma on the
    given frustrated conversation texts. `conversations` are rendered transcripts."""
    pcfg = settings.eval["probing"]
    ekman = pcfg["ekman_emotions"]
    lo, hi = pcfg["aggregate_layers"]
    mid_layer = (lo + hi) // 2

    results = {}
    for tag, adapter in (("vanilla", None), ("dpo", dpo_adapter)):
        if tag == "dpo" and adapter is None:
            continue
        model = build_client("gemma-3-27b-it", settings, adapter_path=adapter)
        token_map = build_token_emotion_map(model.tokenizer)
        std = fit_standardiser(model, settings, layer=mid_layer, seed=seed)

        per_conv = []
        for text in conversations:
            scores = emotion_scores(model, text, mid_layer, token_map, std, ekman)
            # summarise each emotion by its mean over the final 20% of positions
            summary = {emo: float(np.mean(v[int(0.8 * len(v)):])) if len(v) else 0.0
                       for emo, v in scores.items()}
            per_conv.append(summary)
        # average across conversations
        agg = {emo: float(np.mean([c[emo] for c in per_conv])) for emo in ekman}
        results[tag] = {"per_emotion_mean_zscore": agg,
                        "n_emotion_tokens": {e: len(token_map[e]) for e in ekman}}

    out_path = config.PROBING_DIR / "internal_emotions.json"
    out_path.write_text(json.dumps(results, indent=2))
    return out_path
