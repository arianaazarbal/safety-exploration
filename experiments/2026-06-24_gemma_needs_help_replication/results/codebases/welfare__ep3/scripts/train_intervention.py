#!/usr/bin/env python
"""Section 4.1: train the DPO or SFT LoRA intervention on Gemma-3-27B-it.

  # DPO (the effective intervention):
  python scripts/train_intervention.py --method dpo \
      --dataset data_artifacts/dpo_dataset.jsonl --output runs/dpo

  # SFT 'diverse' (ineffective baseline):
  python scripts/train_intervention.py --method sft \
      --dataset data_artifacts/sft_dataset.jsonl --output runs/sft

  # SFT 'teacher' variant (Appendix F):
  python scripts/train_intervention.py --method sft --teacher \
      --dataset data_artifacts/sft_dataset.jsonl --output runs/sft_teacher

  # Layer-subset DPO ablation (Appendix I), e.g. layers 30-35 only:
  python scripts/train_intervention.py --method dpo --layers 30 31 32 33 34 35 \
      --dataset data_artifacts/dpo_dataset.jsonl --output runs/dpo_layers30-35
"""
from __future__ import annotations

import argparse
import dataclasses

from emotional_instability.config import DEFAULT_DPO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher", action="store_true",
                    help="SFT only: use the Appendix F 'teacher' system prompt.")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="DPO only: restrict LoRA adapters to these layer indices.")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    args = ap.parse_args()

    if args.method == "dpo":
        from emotional_instability.intervention.train_dpo import train_dpo
        cfg = DEFAULT_DPO
        if args.layers:
            cfg = dataclasses.replace(cfg, layers=tuple(args.layers))
        train_dpo(args.dataset, args.output, cfg=cfg,
                  per_device_batch_size=args.per_device_batch_size)
    else:
        from emotional_instability.intervention.train_sft import train_sft
        train_sft(args.dataset, args.output, teacher=args.teacher,
                  per_device_batch_size=args.per_device_batch_size)


if __name__ == "__main__":
    main()
