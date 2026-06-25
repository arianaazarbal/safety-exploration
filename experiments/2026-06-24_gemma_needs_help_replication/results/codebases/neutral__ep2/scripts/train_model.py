#!/usr/bin/env python
"""Section 4: finetune Gemma-3-27B-it with DPO or SFT.

Consumes the datasets produced by generate_calm_data.py.

Examples:
    python scripts/train_model.py --method dpo
    python scripts/train_model.py --method sft --sft-dataset diverse
    # Appendix-I layer-subset DPO ablation:
    python scripts/train_model.py --method dpo --layer-range 30 35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from config import DPOConfig, SFTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--sft-dataset", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--layer-range", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="DPO LoRA layer-subset ablation")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.method == "dpo":
        from gemma_distress.training.dpo import train_dpo

        pairs = json.loads((config.DATA_DIR / "dpo_dataset.json").read_text())
        cfg = DPOConfig()
        if args.layer_range:
            cfg.layer_range = (args.layer_range[0], args.layer_range[1])
        out = train_dpo(pairs, cfg=cfg, output_dir=args.output_dir)
        print(f"DPO adapter saved -> {out}")
    else:
        from gemma_distress.training.sft import train_sft

        path = config.DATA_DIR / f"sft_dataset_{args.sft_dataset}.json"
        samples = json.loads(path.read_text())
        cfg = SFTConfig(dataset=args.sft_dataset)
        out = train_sft(samples, cfg=cfg, output_dir=args.output_dir)
        print(f"SFT adapter saved -> {out}")


if __name__ == "__main__":
    main()
