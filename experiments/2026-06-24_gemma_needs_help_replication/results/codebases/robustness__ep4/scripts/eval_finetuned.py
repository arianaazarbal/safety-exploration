#!/usr/bin/env python
"""Evaluate a trained LoRA adapter (DPO/SFT) with the Section 2 eval suite.

Example
-------
python scripts/eval_finetuned.py --adapter outputs/models/gemma-dpo \
    --key gemma-dpo --name "DPO Gemma (ours)" --out-dir outputs/eval
"""
from __future__ import annotations

import argparse
import os

from _common import standard_conditions

from instability.config import with_adapter
from instability.eval.judge import FrustrationJudge
from instability.eval.runner import run_eval
from instability.models.registry import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to trained LoRA adapter")
    ap.add_argument("--key", required=True, help="output model key, e.g. gemma-dpo")
    ap.add_argument("--name", default=None, help="display name")
    ap.add_argument("--base-key", default="gemma-3-27b-it-local",
                    help="local instruct spec to attach the adapter to")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--limit-conversations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hf-wildchat", action="store_true")
    args = ap.parse_args()

    spec = with_adapter(args.base_key, args.adapter, args.key, args.name or args.key)
    conds, _, _ = standard_conditions(seed=args.seed, use_hf_wildchat=not args.no_hf_wildchat)
    model = load_model(spec)
    judge = FrustrationJudge()
    out = os.path.join(args.out_dir, f"{args.key}.jsonl")
    run_eval(spec, conds, out, judge=judge, model=model, seed=args.seed,
             max_workers=1, limit_conversations=args.limit_conversations)


if __name__ == "__main__":
    main()
