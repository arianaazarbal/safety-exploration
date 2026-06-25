#!/usr/bin/env python
"""Appendix I: compare internal (logit-lens) emotions in vanilla vs DPO Gemma.

Loads a set of high-frustration responses (from Section 2 results) and a
WildChat calibration corpus, then measures internal Ekman-emotion z-scores in
both the vanilla instruct model and the DPO finetune on the SAME texts.
"""
import argparse
import os

from gemma_distress import analysis, config
from gemma_distress.internal_emotion import compare_internal_emotions
from gemma_distress.models import HFChatClient
from gemma_distress.wildchat import load_wildchat_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="DPO LoRA adapter path.")
    ap.add_argument("--n-frustrated", type=int, default=12)
    args = ap.parse_args()

    # High-frustration responses from the vanilla model's Section 2 results.
    recs = analysis.load_records(os.path.join(config.RESULTS_DIR, "section2_gemma-3-27b-it.jsonl"))
    frustrated = [r["response"] for r in sorted(recs, key=lambda r: -r["rating"])
                  if r["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD][: args.n_frustrated]
    wildchat = load_wildchat_prompts(n_prompts=20)

    base_id = config.GEMMA_INSTRUCT["gemma-3-27b-it"]
    vanilla = HFChatClient(base_id)
    dpo = HFChatClient(base_id, adapter_path=args.adapter)

    results = compare_internal_emotions(vanilla, dpo, frustrated, wildchat)
    print("[internal] vanilla:", results["vanilla"])
    print("[internal] dpo    :", results["dpo"])


if __name__ == "__main__":
    main()
