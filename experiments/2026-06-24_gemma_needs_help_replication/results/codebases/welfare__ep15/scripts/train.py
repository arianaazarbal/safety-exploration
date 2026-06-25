#!/usr/bin/env python
"""Section 4.1: train the DPO or SFT LoRA adapter on Gemma-3-27B-it.

    python scripts/train.py dpo
    python scripts/train.py sft
    python scripts/train.py dpo --layers 30 31 32 33 34   # Appendix I ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.dpo.train import train_dpo, train_sft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict LoRA to these layer indices (Appendix I)")
    ap.add_argument("--out", default=None, help="adapter output dir")
    args = ap.parse_args()

    out = Path(args.out) if args.out else None
    if args.method == "dpo":
        train_dpo(output_dir=out, layer_subset=args.layers)
    else:
        train_sft(output_dir=out)


if __name__ == "__main__":
    main()
