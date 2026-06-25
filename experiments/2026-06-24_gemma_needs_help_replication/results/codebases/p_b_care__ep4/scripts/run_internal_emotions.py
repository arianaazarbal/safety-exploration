#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection, vanilla vs DPO Gemma.

Builds the Ekman emotion-token dictionary, fits per-token logit baselines over
WildChat assistant text, then measures per-layer negative-emotion z-scores on
high-frustration conversations for both the vanilla instruct model and the DPO
finetune -- evidence for whether DPO suppresses internal (not just expressed)
emotion.

    python scripts/run_internal_emotions.py
"""
from __future__ import annotations

import json

from emotional_instability.config import ensure_dirs, load_config, model_entry
from emotional_instability.models import get_client
from emotional_instability.models.base import GenerationConfig
from emotional_instability.prompts.wildchat import load_wildchat_prompts
from emotional_instability.internal import emotion_logits as E
from emotional_instability.prefill.seeds import select_seeds


def _baseline_messages(client, n: int):
    """Generate single-turn WildChat assistant responses for baseline stats."""
    prompts = load_wildchat_prompts(min(n, 20), seed=0)
    msgs = []
    gen = GenerationConfig(temperature=1.0, max_new_tokens=256)
    # cycle prompts to reach n samples
    while len(msgs) < n:
        for p in prompts:
            history = [{"role": "user", "content": p["prompt"]}]
            resp = client.chat(history, gen)
            msgs.append(history + [{"role": "assistant", "content": resp}])
            if len(msgs) >= n:
                break
    return msgs


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    lo, hi = cfg.internal_emotions.aggregate_layers
    layers = list(range(lo, hi + 1))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_entry(cfg, "gemma-3-27b-it")["model_id"])
    emo_ids = E.build_emotion_token_ids(tok)
    all_emo_ids = sorted({t for ids in emo_ids.values() for t in ids})
    rand_ids = E.random_token_ids(tok, 1000, seed=cfg.seed)

    results = {}
    for model_name in ("gemma-3-27b-it", "gemma-3-27b-dpo"):
        client = get_client(cfg, model_name)
        n_base = cfg.internal_emotions.wildchat_baseline_samples
        base_msgs = _baseline_messages(client, n_base)
        emo_baseline = E.fit_baseline(client, base_msgs, all_emo_ids, layers)
        rand_baseline = E.fit_baseline(client, base_msgs, rand_ids, layers)

        # high-frustration seed conversations (mined from Section 2)
        seeds = select_seeds(cfg, n_numeric=6, n_text=6, min_score=7, seed=cfg.seed)
        per_seed = []
        for s in seeds:
            traj = E.emotion_trajectory(
                client, s.history, s.final_turn_text, emo_ids,
                emo_baseline, rand_ids, rand_baseline)
            per_seed.append({"seed_id": s.seed_id, "trajectory": traj["scores"]})
        results[model_name] = {"layers": layers, "per_seed": per_seed}

    out = cfg.get_path("artifacts") / "internal_emotions.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
