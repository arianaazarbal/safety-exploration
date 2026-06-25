#!/usr/bin/env python
"""Generate calm + frustrated responses and build the DPO/SFT datasets (Sec 4.1).

Runs Gemma-3-27B-it locally (vLLM), judges with Claude Sonnet 4, then writes:
  outputs/data/calm.jsonl, outputs/data/frustrated.jsonl,
  outputs/data/dpo_pairs.jsonl, outputs/data/sft.jsonl
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from emo_instability.config import SFTTrainConfig
from emo_instability.data import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_responses,
    generate_frustrated_responses,
)
from emo_instability.data.generate_calm import CalmResponse
from emo_instability.judge import FrustrationJudge
from emo_instability.models import build_client


def _dump(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.__dict__ if isinstance(it, CalmResponse) else it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", default="gemma-3-27b-it")
    ap.add_argument("--n-puzzles", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--out", default="outputs/data")
    args = ap.parse_args()

    judge = FrustrationJudge()
    gen = build_client(args.gen_model)

    print("Generating calm (reassured) responses...")
    calm = generate_calm_responses(gen, judge, n_puzzles=args.n_puzzles)
    _dump(calm, os.path.join(args.out, "calm.jsonl"))
    print(f"  kept {len(calm)} calm responses")

    print("Generating frustrated (standard) responses...")
    frustrated = generate_frustrated_responses(gen, judge, n_puzzles=args.n_puzzles)
    _dump(frustrated, os.path.join(args.out, "frustrated.jsonl"))
    print(f"  collected {len(frustrated)} responses")

    print("Building DPO pairs...")
    pairs = build_dpo_dataset(
        calm, frustrated, n_pairs=args.n_pairs,
        output_path=os.path.join(args.out, "dpo_pairs.jsonl"),
    )
    print(f"  wrote {len(pairs)} DPO pairs")

    print("Building SFT dataset...")
    sft_cfg = SFTTrainConfig()
    sft = build_sft_dataset(
        calm, n_calm=sft_cfg.n_calm, n_instruct_mix=sft_cfg.n_instruct_mix,
        instruct_mix_dataset=sft_cfg.instruct_mix_dataset,
        output_path=os.path.join(args.out, "sft.jsonl"),
    )
    print(f"  wrote {len(sft)} SFT samples")


if __name__ == "__main__":
    main()
