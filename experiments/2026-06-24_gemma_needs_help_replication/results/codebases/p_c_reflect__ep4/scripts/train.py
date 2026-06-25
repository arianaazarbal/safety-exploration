#!/usr/bin/env python
"""Section 4.1: finetune Gemma-3-27B-it with SFT or DPO (LoRA).

Requires calm data (scripts/generate_calm_data.py) and, for DPO, Section 2
transcripts for gemma-3-27b-it (the source of frustrated responses).

    python scripts/train.py dpo                       # DPO, all layers
    python scripts/train.py dpo --layers 30 35        # Appendix I ablation
    python scripts/train.py sft                       # diverse SFT
    python scripts/train.py sft --teacher             # teacher SFT
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=["sft", "dpo"])
    p.add_argument("--layers", nargs=2, type=int, default=None,
                   help="DPO LoRA layer window, e.g. --layers 30 35")
    p.add_argument("--teacher", action="store_true", help="SFT teacher variant")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.method == "dpo":
        from gemma_distress.training.dpo import train_dpo

        layers = tuple(args.layers) if args.layers else None
        out = train_dpo(output_dir=args.output, layers=layers, seed=args.seed)
    else:
        from gemma_distress.training.sft import train_sft

        out = train_sft(output_dir=args.output, teacher=args.teacher, seed=args.seed)
    print(f"Saved adapter to: {out}")


if __name__ == "__main__":
    main()
