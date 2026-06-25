#!/usr/bin/env python
"""LoRA SFT / DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Examples
--------
python scripts/train.py --method dpo --data outputs/training_data/dpo.jsonl \
    --output outputs/adapters/dpo
python scripts/train.py --method sft --data outputs/training_data/sft.jsonl \
    --output outputs/adapters/sft

# Appendix I layer ablation (LoRA on layers 30-35 only):
python scripts/train.py --method dpo --data outputs/training_data/dpo.jsonl \
    --output outputs/adapters/dpo_L30-35 --layers 30 31 32 33 34
"""

from __future__ import annotations

import argparse

from emotional_instability.training.train import dpo_config, sft_config, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these decoder-layer indices (App I)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    maker = sft_config if args.method == "sft" else dpo_config
    cfg = maker(
        base_model=args.base_model,
        output_dir=args.output,
        layers=args.layers,
        seed=args.seed,
    )
    out = train(cfg, args.data)
    print(f"[train] saved adapter -> {out}")


if __name__ == "__main__":
    main()
