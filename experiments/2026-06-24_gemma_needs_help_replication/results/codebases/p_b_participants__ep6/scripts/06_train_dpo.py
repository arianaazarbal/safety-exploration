#!/usr/bin/env python
"""Section 4: DPO finetune Gemma-3-27B-it on the 280 preference pairs.

Usage:
    python scripts/06_train_dpo.py
    python scripts/06_train_dpo.py --layers 30 31 32 33 34   # Appendix I ablation
"""
from pathlib import Path

from _common import base_parser, cfg_from_args

from emotional_instability.training.dpo import train_dpo


def main():
    p = base_parser(__doc__)
    p.add_argument("--pairs", default=None, help="DPO pairs jsonl (default: runs/training/dpo_pairs.jsonl)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="restrict LoRA to these layer indices (Appendix I); default all layers")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    if args.layers:
        cfg["dpo"]["layers"] = args.layers
    pairs = args.pairs or str(Path(cfg["run"]["output_dir"]) / "training" / "dpo_pairs.jsonl")
    out = train_dpo(cfg, pairs)
    print(f"DPO adapter saved to {out}")


if __name__ == "__main__":
    main()
