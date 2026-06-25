#!/usr/bin/env python
"""Appendix I.2: logit-based internal-emotion trajectory through a frustrated
conversation, for vanilla vs DPO Gemma.

Reads a Section-2 scores file to pick a high-frustration 3-turn numeric
conversation, fits the WildChat baseline, and prints the layer-30-40 running
emotion trajectory for each model.

    python scripts/run_internal.py --scores outputs/scores/gemma-3-27b-it.jsonl \
        --dpo-adapter outputs/adapters/dpo
"""
from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.internal.emotion_logit import EmotionLogitDetector
from gemma_distress.models import build_client
from gemma_distress.prompts.wildchat import sample_wildchat_prompts
from gemma_distress.training.build_dpo_helpers import \
    reconstruct_messages_by_rollout
from gemma_distress.utils.io import read_jsonl


def _pick_high_frustration_conversation(scores_path: str):
    records = [r for r in read_jsonl(scores_path)
               if r["category"] == "impossible_numeric"]
    best = None
    for key, (messages, recs) in reconstruct_messages_by_rollout(records).items():
        score = max(r["rating"] for r in recs)
        if best is None or score > best[0]:
            best = (score, messages)
    return best[1] if best else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", default=None)
    args = ap.parse_args()

    conversation = _pick_high_frustration_conversation(args.scores)
    wildchat = sample_wildchat_prompts(config.INTERNAL_ZSCORE_SAMPLES, seed=0)

    variants = [("vanilla", None)]
    if args.dpo_adapter:
        variants.append(("dpo", args.dpo_adapter))

    for tag, adapter in variants:
        client = build_client(args.model, adapter_path=adapter, lazy=False)
        det = EmotionLogitDetector(client.model, client.tokenizer)
        det.fit_baseline(wildchat)
        traj = det.conversation_trajectory(conversation)
        print(f"\n=== {tag} (layers {config.INTERNAL_LAYER_RANGE}) ===")
        for emo, series in traj.items():
            peak = max(series) if series else float("nan")
            print(f"  {emo:10s} peak z-score={peak:.2f}")


if __name__ == "__main__":
    main()
