"""Train an SFT LoRA adapter on Gemma-3-27B-it (Section 4.1 / Appendix E/F).

Example:
    distress-train-sft --dataset runs/training_data/sft_dataset --out runs/adapters/sft
"""

from __future__ import annotations

import argparse
import dataclasses

from datasets import load_from_disk

from ..config import SFT
from ..training.sft import train_sft
from ._common import out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT finetuning of Gemma-3-27B-it.")
    ap.add_argument("--dataset", required=True, help="path to saved SFT dataset")
    ap.add_argument("--out", default=str(out_dir("adapters") / "sft"))
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse",
                    help="which calming dataset (Appendix F)")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = dataclasses.replace(SFT, variant=args.variant)
    ds = load_from_disk(args.dataset)
    path = train_sft(ds, args.out, cfg=cfg, per_device_batch_size=args.per_device_batch_size)
    print(f"Saved SFT ({args.variant}) adapter -> {path}")


if __name__ == "__main__":
    main()
