"""Section 4.1: train the DPO and/or SFT LoRA adapters on Gemma-3-27B-it.

Usage:
    python scripts/04_train.py --method dpo
    python scripts/04_train.py --method sft --variant diverse
    python scripts/04_train.py --method dpo --layers 30 35   # Appendix I ablation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHECKPOINTS_DIR, DATA_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--variant", default="diverse", help="SFT calm-data variant")
    ap.add_argument("--layers", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict LoRA to decoder layers [LO, HI) (DPO ablation)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    load_in_4bit = not args.no_4bit

    if args.method == "dpo":
        from src.training.train_dpo import train_dpo

        suffix = f"_L{args.layers[0]}-{args.layers[1]}" if args.layers else ""
        out = train_dpo(
            DATA_DIR / "dpo_pairs.jsonl",
            output_dir=CHECKPOINTS_DIR / f"dpo_gemma27b{suffix}",
            target_layers=tuple(args.layers) if args.layers else None,
            load_in_4bit=load_in_4bit,
        )
    else:
        from src.training.train_sft import train_sft

        out = train_sft(
            DATA_DIR / f"sft_{args.variant}.jsonl",
            output_dir=CHECKPOINTS_DIR / f"sft_{args.variant}_gemma27b",
            variant=args.variant,
            load_in_4bit=load_in_4bit,
        )
    print(f"[done] adapter saved to {out}")


if __name__ == "__main__":
    main()
