#!/usr/bin/env python
"""Section 4: LoRA fine-tune Gemma-3-27B-it with DPO or SFT.

Examples
--------
python scripts/run_finetune.py --method dpo
python scripts/run_finetune.py --method sft
# Appendix I layer ablation (adapters on layers 30-35 only):
python scripts/run_finetune.py --method dpo --layers 30 31 32 33 34 35 \
    --output-name dpo_gemma_l30-35
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import (  # noqa: E402
    BASE_FINETUNE_MODEL, DATA_DIR, DPO_CONFIG, MODELS, SFT_CONFIG,
)
from emoeval.train import train_dpo, train_sft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--base-model", default=BASE_FINETUNE_MODEL, choices=list(MODELS))
    ap.add_argument("--data", default=None, help="Override training data path.")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="Restrict LoRA adapters to these decoder layer indices (Appendix I).")
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    if args.method == "dpo":
        cfg = DPO_CONFIG
        data = args.data or os.path.join(DATA_DIR, "dpo_pairs.jsonl")
        out_name = args.output_name or "dpo_gemma"
    else:
        cfg = SFT_CONFIG
        data = args.data or os.path.join(DATA_DIR, "sft_data.jsonl")
        out_name = args.output_name or "sft_gemma"

    if args.layers:
        cfg = dataclasses.replace(cfg, layers_to_train=tuple(args.layers))

    print(f"Fine-tuning {args.base_model} with {args.method.upper()} "
          f"(layers={'all' if not args.layers else args.layers}) ...")
    if args.method == "dpo":
        out = train_dpo(cfg, data, args.base_model, out_name)
    else:
        out = train_sft(cfg, data, args.base_model, out_name)
    print(f"Saved adapter -> {out}")


if __name__ == "__main__":
    main()
