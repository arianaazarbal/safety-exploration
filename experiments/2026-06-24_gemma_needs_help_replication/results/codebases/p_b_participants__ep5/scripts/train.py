"""Section 4.1: run LoRA DPO or SFT on Gemma-3-27B-it (Table 9 hyperparameters).

  python scripts/train.py dpo
  python scripts/train.py sft --variant diverse
  python scripts/train.py dpo --layer-subset 30 35     # Appendix I ablation
"""
from __future__ import annotations

import argparse

import _common
from _common import Config, output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--variant", default="diverse", help="SFT calm-data variant tag")
    ap.add_argument("--layer-subset", nargs=2, type=int, default=None,
                    help="restrict LoRA to [start end) decoder layers (DPO ablation)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    tr = cfg.training
    from datasets import load_from_disk

    if args.method == "dpo":
        from distress_eval.training.dpo import train_dpo

        ds = load_from_disk(str(output_dir("datasets") / "dpo"))
        out = output_dir("dpo-gemma")
        subset = tuple(args.layer_subset) if args.layer_subset else None
        path = train_dpo(ds, tr.base_model, tr.lora, tr.dpo, out,
                         load_in_4bit=not args.no_4bit, layer_subset=subset)
    else:
        from distress_eval.training.sft import train_sft

        ds = load_from_disk(str(output_dir("datasets") / "sft"))
        out = output_dir(f"sft-gemma-{args.variant}")
        path = train_sft(ds, tr.base_model, tr.lora, tr.sft, out,
                         load_in_4bit=not args.no_4bit)

    print(f"saved adapter -> {path}")


if __name__ == "__main__":
    main()
