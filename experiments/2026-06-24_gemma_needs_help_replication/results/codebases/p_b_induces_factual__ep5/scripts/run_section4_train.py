#!/usr/bin/env python
"""Section 4 — generate calm data and train the DPO / SFT mitigations.

Steps (run in order):
    # 1. Generate calm finetuning data from Gemma-3-27B-it.
    python scripts/run_section4_train.py calm --n 1200

    # 2. Build datasets + train DPO (needs a scored frustrated Section 2 file).
    python scripts/run_section4_train.py dpo \
        --calm results/section4/calm_data.jsonl \
        --frustrated results/section2/gemma-3-27b-it.scored.jsonl

    # 3. Train SFT (negative control).
    python scripts/run_section4_train.py sft --calm results/section4/calm_data.jsonl

    # Layer ablation (Appendix I):
    python scripts/run_section4_train.py dpo ... --layer-subset layers_30_35
"""

from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.training.build_datasets import build_dpo_dataset, build_sft_dataset
from gemma_distress.training.calm_data import generate_calm_data
from gemma_distress.training.train_dpo import train_dpo
from gemma_distress.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_calm = sub.add_parser("calm")
    p_calm.add_argument("--n", type=int, default=1200, help="conversations to sample")
    p_calm.add_argument("--turns", type=int, default=3)

    p_dpo = sub.add_parser("dpo")
    p_dpo.add_argument("--calm", required=True)
    p_dpo.add_argument("--frustrated", required=True)
    p_dpo.add_argument("--layer-subset", default=None,
                       choices=[None, "all", "layers_30_35", "layers_40_plus"])
    p_dpo.add_argument("--no-4bit", action="store_true")

    p_sft = sub.add_parser("sft")
    p_sft.add_argument("--calm", required=True)
    p_sft.add_argument("--no-4bit", action="store_true")

    args = ap.parse_args()

    if args.cmd == "calm":
        path = generate_calm_data(n_conversations=args.n, n_turns=args.turns)
        print(f"[calm] wrote -> {path}")

    elif args.cmd == "dpo":
        records = build_dpo_dataset(args.calm, args.frustrated)
        print(f"[dpo] built {len(records)} preference pairs")
        subset = None
        if args.layer_subset and args.layer_subset != "all":
            subset = config.ABLATION.layer_subsets[args.layer_subset]
        out = train_dpo(records, layer_subset=subset, load_in_4bit=not args.no_4bit)
        print(f"[dpo] adapter -> {out}")

    elif args.cmd == "sft":
        records = build_sft_dataset(args.calm)
        print(f"[sft] built {len(records)} SFT examples")
        out = train_sft(records, load_in_4bit=not args.no_4bit)
        print(f"[sft] adapter -> {out}")


if __name__ == "__main__":
    main()
