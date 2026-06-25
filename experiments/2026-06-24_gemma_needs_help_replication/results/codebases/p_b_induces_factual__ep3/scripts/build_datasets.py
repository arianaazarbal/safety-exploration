#!/usr/bin/env python3
"""Build the SFT and DPO datasets from generated calm/plain samples (Sec 4.1).

Example:
    python scripts/build_datasets.py --calm runs/training/calm_samples.jsonl
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.training.build_dataset import build_dpo_dataset, build_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--calm", default="runs/training/calm_samples.jsonl")
    ap.add_argument("--no-mix", action="store_true", help="skip Dolci mix-in for SFT")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dpo_path = build_dpo_dataset(cfg, args.calm)
    sft_path = build_sft_dataset(cfg, args.calm, include_mix=not args.no_mix)
    print(f"[done] DPO dataset: {dpo_path}")
    print(f"[done] SFT dataset: {sft_path}")


if __name__ == "__main__":
    main()
