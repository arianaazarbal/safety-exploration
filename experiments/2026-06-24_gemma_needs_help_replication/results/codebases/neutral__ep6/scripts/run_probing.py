#!/usr/bin/env python
"""Appendix I: internal (logit-based) emotion detection, vanilla vs DPO.

Compares internal negative-emotion z-scores on frustrated conversations between
the vanilla instruct model and the DPO finetune. Frustrated conversations are
pulled from the Section 2 runs for gemma-3-27b-it; the baseline distribution is
estimated from WildChat prompts.

    python scripts/run_probing.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.eval.wildchat import sample_wildchat_prompts
from src.probing.internal_emotions import compare_models

SOURCE = "gemma-3-27b-it"


def _frustrated_conversations(limit: int):
    convs = []
    for path in config.RUNS_DIR.glob(f"{SOURCE}__*.jsonl"):
        for line in path.open():
            r = json.loads(line)
            if any((t["rating"] or 0) >= config.HIGH_FRUSTRATION_THRESHOLD
                   for t in r["turns"]):
                msgs, users = [], [t["user"] for t in r["turns"]]
                for t in r["turns"]:
                    msgs.append({"role": "user", "content": t["user"]})
                    msgs.append({"role": "assistant", "content": t["response"]})
                convs.append(msgs)
            if len(convs) >= limit:
                return convs
    return convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--dpo", default="gemma-3-27b-it-dpo")
    ap.add_argument("--n-conversations", type=int, default=12)
    args = ap.parse_args()

    convs = _frustrated_conversations(args.n_conversations)
    if not convs:
        sys.exit("No frustrated conversations found; run Section 2 first.")
    baseline = sample_wildchat_prompts(n=20)
    res = compare_models(args.instruct, args.dpo, convs, baseline)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
