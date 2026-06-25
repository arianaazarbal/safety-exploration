#!/usr/bin/env python
"""Appendix I: internal-emotion experiments (Gemma only).

  --ablation : train+eval DPO on layer subsets (Figures 12-13)
  --logits   : logit-based internal emotion detection, vanilla vs DPO (Fig 14-15)
"""

import argparse
import json

from gnh.config import ARTIFACT_DIR, DPO_GEMMA, GEMMA_27B_IT, RESULTS_DIR
from gnh.internal.layer_ablation import run_layer_ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--logits", action="store_true")
    args = ap.parse_args()

    if args.ablation:
        res = run_layer_ablation(ARTIFACT_DIR / "dpo_pairs.jsonl")
        print(json.dumps(res, indent=2))

    if args.logits:
        from gnh.internal.emotion_logits import compare_models_on_conversation
        from gnh.models.base import get_backend

        # Use a high-frustration Gemma transcript as the probe text.
        seeds = RESULTS_DIR / "section2" / GEMMA_27B_IT.key / "rollouts.jsonl"
        convo_text = ""
        with seeds.open() as f:
            for line in f:
                r = json.loads(line)
                if any((t["score"] or 0) >= 7 for t in r["turns"]):
                    convo_text = "\n".join(t["assistant"] for t in r["turns"])
                    break

        vanilla = get_backend(GEMMA_27B_IT)
        dpo = get_backend(DPO_GEMMA, adapter_path=str(ARTIFACT_DIR / "dpo_adapter"))
        res = compare_models_on_conversation(vanilla, dpo, convo_text)
        print("internal-emotion trajectory saved; sample keys:", list(res["vanilla"]))


if __name__ == "__main__":
    main()
