#!/usr/bin/env python
"""Section 4: DPO finetuning of Gemma-3-27B-it (LoRA).

    python scripts/run_dpo.py --qlora
    python scripts/run_dpo.py --layers 30 31 32 33 34   # Appendix I ablation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.train_dpo import train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlora", action="store_true", help="4-bit QLoRA (single GPU)")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (Appendix I)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    train_dpo(output_dir=Path(args.output_dir) if args.output_dir else None,
              qlora=args.qlora, layers_to_transform=args.layers)


if __name__ == "__main__":
    main()
