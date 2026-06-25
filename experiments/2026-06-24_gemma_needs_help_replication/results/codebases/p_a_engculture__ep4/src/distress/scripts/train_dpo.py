"""Train the DPO LoRA adapter on Gemma-3-27B-it (Section 4.1).

Example:
    distress-train-dpo --dataset runs/training_data/dpo_dataset --out runs/adapters/dpo
    # layer ablation (Appendix I):
    distress-train-dpo --dataset ... --out runs/adapters/dpo_ablation --layer-ablation
"""

from __future__ import annotations

import argparse

from datasets import load_from_disk

from ..config import DPO
from ..training.dpo import train_dpo
from ..training.layer_ablation import run_layer_ablation
from ._common import out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO finetuning of Gemma-3-27B-it.")
    ap.add_argument("--dataset", required=True, help="path to saved DPO dataset")
    ap.add_argument("--out", default=str(out_dir("adapters") / "dpo"))
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--layer-ablation", action="store_true",
                    help="train one adapter per layer subset (Appendix I)")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    if args.layer_ablation:
        outputs = run_layer_ablation(ds, args.out)
        print("Trained layer-ablation adapters:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    else:
        path = train_dpo(ds, args.out, cfg=DPO, per_device_batch_size=args.per_device_batch_size)
        print(f"Saved DPO adapter -> {path}")


if __name__ == "__main__":
    main()
