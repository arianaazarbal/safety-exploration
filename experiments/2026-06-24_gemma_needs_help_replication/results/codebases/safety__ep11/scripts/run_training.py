"""Section 4: train the DPO and/or SFT LoRA adapters on Gemma-3-27B-it.

Examples:
    python scripts/run_training.py --method dpo
    python scripts/run_training.py --method sft
    # Appendix I layer ablation (adapters on layers 30-35 only):
    python scripts/run_training.py --method dpo --layers 30 35
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import config
from src.training.train_dpo import train_dpo
from src.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--dpo-pairs", type=Path, default=config.ARTIFACT_DIR / "dpo_pairs.jsonl")
    ap.add_argument("--sft-data", type=Path, default=config.ARTIFACT_DIR / "sft_data.jsonl")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("START", "END"),
                    help="restrict DPO LoRA adapters to layers [START, END) (Appendix I)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.method == "dpo":
        layers = range(args.layers[0], args.layers[1]) if args.layers else None
        train_dpo(args.dpo_pairs, output_dir=args.out, layers_to_tune=layers)
    else:
        train_sft(args.sft_data, output_dir=args.out)


if __name__ == "__main__":
    main()
