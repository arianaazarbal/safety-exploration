#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data and build SFT/DPO datasets.

Produces paired (vanilla, calm) conversations on impossible numeric puzzles,
then materialises the SFT and DPO datasets to disk.
"""
import argparse

import _bootstrap  # noqa: F401
import config
from src.training import generate_calm_data, build_sft_dataset, build_dpo_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()

    generate_calm_data(variant=args.variant)
    sft_ds = build_sft_dataset(variant=args.variant)
    sft_ds.save_to_disk(str(config.DATA_DIR / f"sft_dataset_{args.variant}"))
    if args.variant == "diverse":
        dpo_ds = build_dpo_dataset()
        dpo_ds.save_to_disk(str(config.DATA_DIR / "dpo_dataset"))
        print(f"DPO pairs: {len(dpo_ds)}")
    print(f"SFT examples: {len(sft_ds)}")


if __name__ == "__main__":
    main()
