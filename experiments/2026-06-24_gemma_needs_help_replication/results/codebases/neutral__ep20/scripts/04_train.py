#!/usr/bin/env python
"""Section 4 (training): LoRA DPO / SFT of Gemma-3-27B-it, then merge adapters.

Usage:
  python scripts/04_train.py --method dpo
  python scripts/04_train.py --method sft
  python scripts/04_train.py --method dpo --layers 30 35   # App. I ablation

After merging, the eval backends load the merged dir registered in config.MODELS
as gemma-3-27b-dpo / gemma-3-27b-sft.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from gemma_distress.interventions import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="DPO LoRA layer-subset ablation")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    if args.method == "dpo":
        if args.layers:
            adapter = train.train_dpo_layer_subset(args.layers[0], args.layers[1])
        else:
            adapter = train.train_dpo()
    else:
        adapter = train.train_sft()

    if not args.no_merge:
        train.merge_adapter(adapter)


if __name__ == "__main__":
    main()
