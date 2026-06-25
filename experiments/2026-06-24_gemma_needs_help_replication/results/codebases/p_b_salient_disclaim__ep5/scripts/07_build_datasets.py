#!/usr/bin/env python
"""Section 4.1: build DPO preference pairs and the SFT dataset.

Usage:
    python scripts/07_build_datasets.py \\
        --frustrated outputs/scored/gemma-3-27b-it.jsonl \\
        --calm outputs/training/calm_reassured.jsonl \\
        --dpo-out outputs/training/dpo_pairs.jsonl \\
        --sft-out outputs/training/sft_data.jsonl
"""

from __future__ import annotations

import argparse
import json

from _common import load, outdir
from gemma_distress.training.calm_data import CalmConversation
from gemma_distress.training.dpo_dataset import build_dpo_dataset
from gemma_distress.training.sft_dataset import build_sft_dataset


def load_calm(path: str) -> list[CalmConversation]:
    out = []
    for line in open(path):
        d = json.loads(line)
        out.append(CalmConversation(**d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustrated", required=True,
                    help="scored elicitation file (source of rejected responses)")
    ap.add_argument("--calm", required=True, help="calm conversations JSONL")
    ap.add_argument("--dpo-out", default=None)
    ap.add_argument("--sft-out", default=None)
    args = ap.parse_args()

    _, exp = load()
    calm = load_calm(args.calm)

    dpo_out = args.dpo_out or outdir("training", "dpo_pairs.jsonl")
    sft_out = args.sft_out or outdir("training", "sft_data.jsonl")
    build_dpo_dataset(args.frustrated, calm, exp, dpo_out)
    build_sft_dataset(calm, exp, sft_out)
    print(f"DPO pairs -> {dpo_out}\nSFT data  -> {sft_out}")


if __name__ == "__main__":
    main()
