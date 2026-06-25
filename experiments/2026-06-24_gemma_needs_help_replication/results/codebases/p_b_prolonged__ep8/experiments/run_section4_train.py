"""Section 4.1: train the DPO and/or SFT LoRA adapters on Gemma-3-27B-it.

Usage:
    python experiments/run_section4_train.py --method dpo
    python experiments/run_section4_train.py --method sft
    python experiments/run_section4_train.py --method dpo --layers 30 35   # Section 4.2 ablation
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import dataclasses

import config
from gemma_needs_help.finetuning.train import train_dpo, train_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA-style 4-bit base")
    ap.add_argument("--layers", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="restrict LoRA to an inclusive layer range (Section 4.2 ablation)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lora = config.LoRAConfig(
        layers_to_transform=tuple(args.layers) if args.layers else None
    )

    if args.method == "dpo":
        cfg = dataclasses.replace(config.DPO, lora=lora)
        train_dpo(str(config.CALM_DATA_DIR / "dpo_pairs.jsonl"),
                  out_dir=args.out_dir, cfg=cfg, load_in_4bit=args.load_in_4bit)
    else:
        cfg = dataclasses.replace(config.SFT, lora=lora)
        train_sft(str(config.CALM_DATA_DIR / "sft_dataset.jsonl"),
                  out_dir=args.out_dir, cfg=cfg, load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    main()
