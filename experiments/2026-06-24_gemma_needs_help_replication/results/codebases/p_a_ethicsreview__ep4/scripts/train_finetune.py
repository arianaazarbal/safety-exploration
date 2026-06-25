#!/usr/bin/env python
"""Train the DPO or SFT LoRA finetune of Gemma-3-27B-it (Section 4.1, Table 9).

python scripts/train_finetune.py --method dpo \
    --data results/finetune_data/dpo_pairs.jsonl --out-dir checkpoints/dpo

For the Appendix I layer ablation, override the LoRA layer range, e.g.:
    --lora-layers 30 35      # adapters on layers 30-34 only

Requires training hardware (a 27B LoRA finetune). Not executed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.training.train import train_dpo, train_sft  # noqa: E402
from emotional_instability.utils.io import load_config, read_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--data", required=True, help="JSONL dataset path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--lora-layers", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="Restrict LoRA to layers [START, END) (Appendix I ablation)")
    args = ap.parse_args()

    cfg = load_config("training")
    seed_everything(cfg.get("seed", 0))
    if args.lora_layers is not None:
        cfg["lora"]["layers"] = list(args.lora_layers)

    data = list(read_jsonl(args.data))
    if args.method == "dpo":
        train_dpo(cfg, data, args.out_dir, args.per_device_batch_size)
    else:
        train_sft(cfg, data, args.out_dir, args.per_device_batch_size)
    print(f"saved {args.method} adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
