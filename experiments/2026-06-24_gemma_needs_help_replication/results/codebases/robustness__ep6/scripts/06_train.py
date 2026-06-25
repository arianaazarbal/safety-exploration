#!/usr/bin/env python
"""Section 4: train the DPO or SFT LoRA adapter on Gemma-3-27B-it.

Examples
--------
python scripts/06_train.py --method dpo
python scripts/06_train.py --method sft
# Appendix I depth ablation: adapters on layers [30,35) only
python scripts/06_train.py --method dpo --layers 30 35
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval.training.train import train_dpo, train_sft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="restrict LoRA adapters to decoder layers [LO, HI)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    if args.method == "dpo":
        config = cfg.DPO_CONFIG
        if args.layers:
            config = replace(config, lora=replace(config.lora,
                                                  layers=tuple(args.layers)))
        train_dpo(config=config, output_dir=out)
    else:
        config = cfg.SFT_CONFIG
        if args.layers:
            config = replace(config, lora=replace(config.lora,
                                                  layers=tuple(args.layers)))
        train_sft(config=config, output_dir=out)


if __name__ == "__main__":
    main()
