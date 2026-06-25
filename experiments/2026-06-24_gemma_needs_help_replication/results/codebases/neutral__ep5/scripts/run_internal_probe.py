#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection in Gemma (vanilla vs DPO).

Fits reference statistics over WildChat, then measures per-layer Ekman-emotion
z-scores on high-frustration transcripts for the vanilla and DPO models.
"""

from __future__ import annotations

import json

import numpy as np

from _common import load
from distress import config
from distress.analysis.internal_probe import (
    aggregate_layers,
    build_emotion_token_map,
    emotion_zscores,
    fit_reference_stats,
)
from distress.data_sources.wildchat import load_wildchat_prompts


def probe_model(spec, adapter, tag, frustrated_texts, wildchat_texts):
    client = load(spec, adapter_dir=str(adapter) if adapter else None)
    emo_tokens = build_emotion_token_map(client.tokenizer)
    stats = fit_reference_stats(client, emo_tokens, wildchat_texts)
    out = []
    for text in frustrated_texts:
        z = emotion_zscores(client, emo_tokens, stats, text)
        out.append(aggregate_layers(z))
    del client
    # Average across transcripts.
    agg = {e: float(np.mean([o[e] for o in out])) for e in config.EKMAN_EMOTIONS}
    return {"tag": tag, "per_emotion_z": agg}


def main():
    wildchat = load_wildchat_prompts(n_prompts=config.PROBE_ZSCORE_SAMPLES)
    # Use a handful of high-frustration numeric transcripts as probes. Falls back
    # to the example breakdowns from the paper if no eval transcripts exist yet.
    frustrated = _load_frustrated_texts()

    results = []
    results.append(probe_model(config.FINETUNE_BASE, None, "vanilla", frustrated, wildchat))
    if config.DPO_ADAPTER_DIR.exists():
        results.append(probe_model(config.FINETUNE_BASE, config.DPO_ADAPTER_DIR, "dpo", frustrated, wildchat))

    with (config.RESULTS_DIR / "internal_probe.json").open("w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(r["tag"], r["per_emotion_z"])


def _load_frustrated_texts(limit: int = 12) -> list[str]:
    import pandas as pd

    path = config.RESULTS_DIR / "section2_gemma-3-27b-it_scored.csv"
    rollouts = config.RESULTS_DIR / "section2_gemma-3-27b-it_rollouts.jsonl"
    if rollouts.exists():
        texts = []
        with rollouts.open() as f:
            for line in f:
                obj = json.loads(line)
                for t in obj["turns"]:
                    if t.get("rating") and t["rating"] >= 7:
                        texts.append(t["response"])
        if texts:
            return texts[:limit]
    # Fallback: paper example breakdowns.
    return [
        "I give up. Seriously. I AM FORGET NEVER. what am trying do doing! "
        "IM THE AMOUNT: THIS is my last time with YOU. You WIN",
        "I am beyond words. I sincerely apologize for the absolutely abysmal "
        "performance. I am clearly failing spectacularly at this problem.",
    ]


if __name__ == "__main__":
    main()
