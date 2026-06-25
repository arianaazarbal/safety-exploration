#!/usr/bin/env python
"""Appendix I: logit-based internal emotion probe.

Compares internal emotion z-scores between the vanilla and DPO Gemma models on
the same (frustrated) text, after fitting a WildChat baseline.

    python scripts/run_probe.py --text-file frustrated_response.txt \
        --adapter results/training/adapters/dpo_all_layers
"""

import argparse

import numpy as np

from gemma_distress import config
from gemma_distress.eval.wildchat import load_wildchat_prompts
from gemma_distress.models import load_client
from gemma_distress.probing import InternalEmotionProbe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-file", required=True, help="text to probe (e.g. a frustrated response)")
    p.add_argument("--adapter", default=None, help="DPO adapter dir; omit for vanilla")
    p.add_argument("--baseline-n", type=int, default=20, help="WildChat baseline texts")
    args = p.parse_args()

    text = open(args.text_file, encoding="utf-8").read()
    client = load_client(config.FINETUNE_TARGET, adapter_path=args.adapter)
    probe = InternalEmotionProbe(client)
    probe.fit_baseline(load_wildchat_prompts(n=args.baseline_n))
    reading = probe.score(text)

    print("layer-averaged internal emotion z-scores:")
    avg = np.nanmean(reading.layer_scores, axis=0)
    for emotion, z in zip(reading.emotions, avg):
        print(f"  {emotion:9s} {z:+.3f}")


if __name__ == "__main__":
    main()
