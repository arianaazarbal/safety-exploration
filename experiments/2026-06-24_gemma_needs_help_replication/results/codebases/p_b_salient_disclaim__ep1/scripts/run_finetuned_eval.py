#!/usr/bin/env python
"""Evaluate a finetuned (LoRA) Gemma variant with the Section 2 suite.

Wraps the elicitation runner so DPO/SFT adapters can be scored exactly like
registry targets, producing the before/after numbers behind Figure 5 (35% ->
0.3%).

Example:
    python scripts/run_finetuned_eval.py \
        --name gemma-3-27b-it-dpo \
        --base-model gemma-3-27b-it \
        --adapter outputs/dpo/adapter \
        --out outputs/elicitation/gemma-3-27b-it-dpo.jsonl
"""
from __future__ import annotations

import argparse
import json

from gemma_distress.config import get_target_spec, register_finetuned_target
from gemma_distress.eval.conditions import build_full_suite
from gemma_distress.eval.metrics import summarise_model
from gemma_distress.eval.runner import run_target
from gemma_distress.models.registry import get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_hf = get_target_spec(args.base_model).params["hf_id"]
    spec = register_finetuned_target(args.name, base_hf, args.adapter)
    client = get_client(spec)

    suite = build_full_suite(seed=args.seed)
    run_target(client, out_path=args.out, suite=suite, seed=args.seed, name=args.name)
    print(json.dumps(summarise_model(args.out), indent=2))


if __name__ == "__main__":
    main()
