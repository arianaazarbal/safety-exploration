#!/usr/bin/env python
"""Section 4.1: LoRA finetune Gemma-3-27B-it (SFT or DPO).

Examples:
    python scripts/08_train.py --method dpo
    python scripts/08_train.py --method sft
    # Section 4.2 layer ablation (adapters on layers 30-35 only):
    python scripts/08_train.py --method dpo --layers 30 31 32 33 34 35 \
        --output outputs/training/dpo_adapter_layers30-35
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict LoRA to these decoder layer indices (Section 4.2)")
    args = ap.parse_args()

    if args.method == "sft":
        from emotional_instability.training.sft import train_sft
        kwargs = dict(output_dir=args.output, lora_rank=args.rank,
                      target_layers=args.layers)
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        if args.lr is not None:
            kwargs["learning_rate"] = args.lr
        out = train_sft(**kwargs)
    else:
        from emotional_instability.training.dpo import train_dpo
        kwargs = dict(output_dir=args.output, lora_rank=args.rank,
                      target_layers=args.layers)
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        if args.lr is not None:
            kwargs["learning_rate"] = args.lr
        out = train_dpo(**kwargs)

    print(f"adapter -> {out}")


if __name__ == "__main__":
    main()
