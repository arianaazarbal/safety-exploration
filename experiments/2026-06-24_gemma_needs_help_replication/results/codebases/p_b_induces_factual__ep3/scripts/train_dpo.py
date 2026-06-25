#!/usr/bin/env python3
"""Train the DPO LoRA finetune of Gemma-3-27B-it (Section 4.1).

Example:
    python scripts/train_dpo.py --data runs/training/dpo.jsonl
    python scripts/train_dpo.py --data runs/training/dpo.jsonl --layers 30 35  # Appendix I
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.training.dpo import train_dpo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="runs/training/dpo.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, nargs=2, default=None,
                    metavar=("START", "END"), help="restrict LoRA to layers [START,END)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = train_dpo(cfg, args.data, output_dir=args.out, layer_subset=args.layers)
    print(f"[done] DPO adapter: {out}")


if __name__ == "__main__":
    main()
