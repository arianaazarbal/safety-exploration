#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma only; Gemini has no
public base model).

Steps:
  1. collect high-frustration seed conversations from Gemma-27B-it
  2. build paraphrased early/onset truncations (Claude labels + paraphrases)
  3. generate + score 50 continuations per prefill for base and instruct Gemma
"""
import argparse
import json
import os

from gemma_distress import config, section3
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import HFBaseClient, HFChatClient
from gemma_distress.onset import OnsetLabeller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["27b", "12b"], default="27b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    it_id = config.GEMMA_INSTRUCT[f"gemma-3-{args.size}-it"]
    pt_id = config.GEMMA_BASE[f"gemma-3-{args.size}-pt"]

    judge = FrustrationJudge()
    labeller = OnsetLabeller()

    print("[section3] loading instruct model + collecting seeds")
    it_client = HFChatClient(it_id)
    seeds = section3.collect_seed_conversations(it_client, judge, seed=args.seed)

    print("[section3] building prefills (onset/early + paraphrase)")
    prefills = section3.build_prefills(seeds, labeller, it_client.tokenizer)
    with open(os.path.join(config.DATA_DIR, "prefills.json"), "w") as f:
        json.dump(prefills, f, indent=2)

    print("[section3] instruct continuations")
    section3.run_continuations(f"gemma-3-{args.size}-it", it_client, prefills, judge)

    print("[section3] loading base model + continuations")
    pt_client = HFBaseClient(pt_id)
    section3.run_continuations(f"gemma-3-{args.size}-pt", pt_client, prefills, judge)
    print("[section3] done")


if __name__ == "__main__":
    main()
