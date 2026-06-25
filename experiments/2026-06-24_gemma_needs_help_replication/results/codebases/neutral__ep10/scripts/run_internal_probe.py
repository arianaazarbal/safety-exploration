#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection.

Compares internal (residual-stream) negative-emotion z-scores between vanilla
Gemma-3-27b-it and the DPO finetune on the same frustrated conversations.

Example:
    python scripts/run_internal_probe.py \
        --conversations results/elicitation/gemma-3-27b-it_rollouts.jsonl \
        --dpo-adapter checkpoints/gemma27b_dpo
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals.prompts import load_wildchat_prompts
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.models.registry import load_model
from emotional_instability.probing.internal_emotions import (
    InternalEmotionDetector, summarise_conversation)


def _conversation_text(rollout) -> str:
    parts = []
    for t in rollout.turns:
        parts.append(t.user_message)
        parts.append(t.assistant_response)
    return "\n".join(parts)


def probe_model(model_name, adapter, conv_texts, wildchat, layers):
    model = load_model(model_name, adapter_path=adapter)
    det = InternalEmotionDetector(model, layers)
    det.fit_baseline(wildchat, n_control=500)
    summaries = []
    for text in conv_texts:
        scores = det.score_text(text)
        summaries.append(summarise_conversation(scores, layers_agg=(30, 40)))
    model.close()
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--n-conversations", type=int, default=12,
                    help="high-frustration conversations to probe (Figure 15 uses 12)")
    ap.add_argument("--layers", nargs="*", type=int,
                    default=list(range(20, 50)))
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "probing"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rollouts = load_rollouts(args.conversations)
    rollouts = sorted(rollouts, key=lambda r: -(r.max_score or 0))[:args.n_conversations]
    conv_texts = [_conversation_text(r) for r in rollouts]
    wildchat = load_wildchat_prompts(20)

    out = {"vanilla": probe_model(config.TARGET_FINETUNE_MODEL, None,
                                  conv_texts, wildchat, args.layers)}
    if args.dpo_adapter:
        out["dpo"] = probe_model(config.TARGET_FINETUNE_MODEL, args.dpo_adapter,
                                 conv_texts, wildchat, args.layers)

    with open(os.path.join(args.out, "internal_emotions.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote internal-emotion trajectories to {args.out}/internal_emotions.json")


if __name__ == "__main__":
    main()
