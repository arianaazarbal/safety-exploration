#!/usr/bin/env python
"""Section 4.1: train the DPO or SFT LoRA finetune of gemma-3-27b-it.

    python scripts/06_train.py --method dpo
    python scripts/06_train.py --method sft
"""
import argparse

import _bootstrap  # noqa: F401
from gemma_distress.training.train import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    args = ap.parse_args()

    if args.method == "dpo":
        out = train_dpo(out_name="dpo/final")
    else:
        out = train_sft(out_name="sft_diverse/final")
    print(f"Saved adapter to {out}")
    print("Register/point the adapter via config/models.yaml to evaluate it "
          "with scripts/01_run_eval.py.")


if __name__ == "__main__":
    main()
