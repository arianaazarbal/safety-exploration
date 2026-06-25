#!/usr/bin/env python
"""Section 4.1: train the DPO and/or SFT LoRA adapters.

Examples
--------
    python scripts/run_section4_train.py --method dpo
    python scripts/run_section4_train.py --method sft --variant diverse
    python scripts/run_section4_train.py --method dpo --layers layers_30_35  # Appendix I
"""
import argparse

import _bootstrap  # noqa: F401
import config
from datasets import load_from_disk
from src.training import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse",
                    help="SFT calm-data variant")
    ap.add_argument("--layers", default="all_layers",
                    choices=list(config.LORA_LAYER_ABLATIONS),
                    help="DPO layer-restricted ablation (Appendix I)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if args.method == "dpo":
        ds = load_from_disk(str(config.DATA_DIR / "dpo_dataset"))
        out = train_dpo(ds, layer_spec_name=args.layers, load_in_4bit=not args.no_4bit)
    else:
        ds = load_from_disk(str(config.DATA_DIR / f"sft_dataset_{args.variant}"))
        out = train_sft(ds, variant=args.variant, load_in_4bit=not args.no_4bit)
    print(f"Saved adapter to {out}")


if __name__ == "__main__":
    main()
