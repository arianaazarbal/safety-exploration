#!/usr/bin/env python
"""Section 4.1: build SFT (1,150-sample) and DPO (280-pair) datasets from the
generated calm/frustrated samples.

  python scripts/build_intervention_datasets.py
"""
from __future__ import annotations

import argparse
import os

from emotional_instability import config
from emotional_instability.intervention.build_datasets import (
    build_dpo_dataset,
    build_sft_dataset,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=config.DATA_DIR)
    args = ap.parse_args()

    calm = os.path.join(args.data_dir, "calm_samples.jsonl")
    frustrated = os.path.join(args.data_dir, "frustrated_samples.jsonl")

    sft_path = build_sft_dataset(calm, out_dir=args.data_dir)
    dpo_path = build_dpo_dataset(calm, frustrated, out_dir=args.data_dir)
    print(f"SFT dataset: {sft_path}")
    print(f"DPO dataset: {dpo_path}")
    print("Next: python scripts/train_intervention.py --method dpo "
          f"--dataset {dpo_path} --output runs/dpo")


if __name__ == "__main__":
    main()
