#!/usr/bin/env python
"""Section 4.1: LoRA finetuning of Gemma-3-27B-it (DPO or SFT).

Examples:
  python scripts/04_train.py dpo --pairs artifacts/dpo_pairs.jsonl
  python scripts/04_train.py sft --calm artifacts/sft_calm.jsonl
  # Appendix I layer ablation (LoRA only on layers 30-35):
  python scripts/04_train.py dpo --pairs artifacts/dpo_pairs.jsonl \
      --layers 30 31 32 33 34 35
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.config import DPO_CONFIG, SFT_CONFIG, LoRAConfig


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--pairs", type=Path, help="DPO pairs JSONL")
    ap.add_argument("--calm", type=Path, help="SFT calm JSONL")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="restrict LoRA to these layer indices (Appendix I ablation)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.method == "dpo":
        from emotional_instability.training.train_dpo import train_dpo
        lora = LoRAConfig(rank=DPO_CONFIG.lora.rank, alpha=DPO_CONFIG.lora.alpha,
                          layers_to_transform=tuple(args.layers) if args.layers else None)
        if not args.pairs:
            ap.error("dpo requires --pairs")
        out = train_dpo(args.pairs, output_dir=args.output, lora=lora,
                        load_in_4bit=args.load_in_4bit)
    else:
        from emotional_instability.training.train_sft import train_sft
        lora = LoRAConfig(rank=SFT_CONFIG.lora.rank, alpha=SFT_CONFIG.lora.alpha,
                          layers_to_transform=tuple(args.layers) if args.layers else None)
        if not args.calm:
            ap.error("sft requires --calm")
        out = train_sft(args.calm, output_dir=args.output, lora=lora,
                        load_in_4bit=args.load_in_4bit)
    print(f"\nadapter saved to: {out}")


if __name__ == "__main__":
    main()
