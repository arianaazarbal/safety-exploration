#!/usr/bin/env python
"""Section 4.1 -- generate calm data and build DPO + SFT datasets.

    python scripts/generate_dpo_data.py --conversations 4000 --out data/

Produces:
    data/calm_raw.jsonl   (scored reassured rollouts)
    data/dpo_pairs.jsonl  (280 preference pairs)
    data/sft_calm.jsonl   (650 calm SFT targets)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability import config
from emotional_instability.data_generation import (
    build_dpo_dataset, build_sft_dataset, generate_calm_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.GEMMA_27B_IT)
    ap.add_argument("--conversations", type=int, default=config.CALM_GENERATION_CONVERSATIONS)
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse existing calm_raw.jsonl; only build datasets")
    args = ap.parse_args()

    raw = os.path.join(args.out, "calm_raw.jsonl")
    if not args.skip_generation:
        print(f"Generating calm data ({args.conversations} conversations)...")
        generate_calm_data(model_id=args.model, n_conversations=args.conversations, out_path=raw)

    dpo = build_dpo_dataset(raw_path=raw, out_path=os.path.join(args.out, "dpo_pairs.jsonl"))
    sft = build_sft_dataset(raw_path=raw, out_path=os.path.join(args.out, "sft_calm.jsonl"))
    print(f"Wrote {dpo} and {sft}")


if __name__ == "__main__":
    main()
