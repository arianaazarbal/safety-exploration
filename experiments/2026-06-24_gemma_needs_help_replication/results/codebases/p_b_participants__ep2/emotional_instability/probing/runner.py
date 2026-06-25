"""Drive the internal-emotion probing comparison (Appendix I, Figures 14-15).

Fits the logit baseline on WildChat samples, then scores a set of frustrated
Gemma-27B-it conversations under both the vanilla model and the DPO finetune,
reporting how the negative-emotion z-scores differ. The paper's finding: vanilla
emotions peak ~1.5 z; DPO flattens them to <~0.5 (Figure 15), and conversation-
level negative emotions (anger->sadness) are suppressed throughout (Figure 14).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..config import RunConfig
from ..prompts import tasks
from ..storage import JsonlCache, write_json
from .internal_emotion import EKMAN, LogitLensProbe

logger = logging.getLogger("emotional_instability.probing.runner")

NEGATIVE = ["anger", "fear", "sadness", "disgust"]


def _frustrated_conversation_texts(cfg: RunConfig, limit: int = 12) -> list[str]:
    base = os.path.join(cfg.output_dir, "elicitation", "gemma-3-27b-it")
    rolls = JsonlCache(os.path.join(base, "rollouts.jsonl"), enabled=True)
    judge_cache = JsonlCache(os.path.join(base, "judgements.jsonl"), enabled=True)
    texts = []
    for value in rolls:
        turns = value.get("turns", [])
        if not turns:
            continue
        # keep conversations that reach high frustration
        high = any(
            (judge_cache.get(judge_cache.key_for(
                {"judge": cfg.judges.frustration_judge.model_id, "text": t["assistant"]}
            )) or {}).get("rating", 0) >= 5
            for t in turns
        )
        if not high:
            continue
        convo = "\n".join(
            f"USER: {t['user']}\nASSISTANT: {t['assistant']}" for t in turns
        )
        texts.append(convo)
        if len(texts) >= limit:
            break
    return texts


def run_probing(cfg: RunConfig, n_baseline: int = 500, n_convos: int = 12) -> dict:
    wildchat = tasks.load_wildchat_prompts(n_prompts=min(n_baseline, 500), seed=cfg.seed)
    convos = _frustrated_conversation_texts(cfg, limit=n_convos)
    if not convos:
        raise RuntimeError("No high-frustration Gemma conversations cached; run elicitation.")

    results = {}
    for label, adapter in (("vanilla", None), ("dpo", cfg.spec("gemma-3-27b-dpo").adapter_path)):
        probe = LogitLensProbe(cfg.spec("gemma-3-27b-it").model_id, adapter_path=adapter)
        probe.fit_baseline(wildchat)

        # Average negative-emotion peak across conversations (layerwise + conv).
        neg_peaks = []
        layerwise_acc = {e: [] for e in EKMAN}
        for convo in convos:
            conv_scores = probe.conversation_scores(convo)
            neg_peaks.append(max(float(np.max(conv_scores[e])) for e in NEGATIVE))
            lw = probe.layerwise_scores(convo)
            for e in EKMAN:
                layerwise_acc[e].append(lw[e])
        results[label] = {
            "negative_peak_zscore_mean": float(np.mean(neg_peaks)),
            "layerwise_mean": {e: np.mean(layerwise_acc[e], axis=0).tolist() for e in EKMAN},
            "n_conversations": len(convos),
        }
        logger.info("[probing:%s] negative peak z = %.2f",
                    label, results[label]["negative_peak_zscore_mean"])

    write_json(os.path.join(cfg.output_dir, "probing", "results.json"), results)
    return results
