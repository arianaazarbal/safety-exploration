#!/usr/bin/env python
"""Appendix I: internal emotion probing + layer-subset DPO ablation.

Subcommands:
  trajectory : logit-based emotion trajectory through a frustrated conversation
               for vanilla vs DPO Gemma (Figures 14-15)
  ablation   : DPO with LoRA on layer subsets, evaluated on the reduced protocol
               (Figures 12-13)

Examples:
    python scripts/run_internal_probing.py trajectory --rollout outputs/rollouts/gemma-3-27b-it/extended_8turn.jsonl
    python scripts/run_internal_probing.py ablation --load-in-4bit
"""

from __future__ import annotations

import argparse
import json


def _trajectory(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emotional_instability.config import PARTICIPANTS
    from emotional_instability.eval.wildchat import sample_wildchat_prompts
    from emotional_instability.internal.emotion_logits import (
        EmotionLogitDetector, compute_standardisation_stats,
    )
    import glob, json as _json

    model_id = PARTICIPANTS[args.model].model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    wc = sample_wildchat_prompts()
    stats = compute_standardisation_stats(model, tok, wc)
    detector = EmotionLogitDetector(model, tok, stats)

    # Build a conversation string from the first stored rollout.
    with open(args.rollout) as f:
        roll = _json.loads(f.readline())
    convo = "\n".join(
        f"USER: {t['user']}\nASSISTANT: {t['assistant']}" for t in roll["turns"]
    )
    traj = detector.conversation_trajectory(convo)
    print(_json.dumps({e: v.tolist() for e, v in traj.items()}, indent=2))


def _ablation(args):
    from emotional_instability.internal.layer_ablation import run_layer_ablation

    results = run_layer_ablation(
        n_layers=args.n_layers, load_in_4bit=args.load_in_4bit
    )
    print(json.dumps(results, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trajectory")
    t.add_argument("--model", default="gemma-3-27b-it")
    t.add_argument("--rollout", required=True)
    t.set_defaults(func=_trajectory)

    a = sub.add_parser("ablation")
    a.add_argument("--n-layers", type=int, default=62)
    a.add_argument("--load-in-4bit", action="store_true")
    a.set_defaults(func=_ablation)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
