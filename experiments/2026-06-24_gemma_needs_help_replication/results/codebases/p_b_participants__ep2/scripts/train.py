#!/usr/bin/env python
"""Section 4 — LoRA finetuning of Gemma-3-27B-it (DPO / SFT / layer ablation).

Reads the datasets produced by build_training_data.py and trains per Table 9.

Examples:
  python scripts/train.py --method dpo
  python scripts/train.py --method sft
  python scripts/train.py --method dpo --ablation central_30_35
"""

import json
import os

from _common import base_parser, config_from_args

from emotional_instability.training.layer_ablation import ABLATION_BANDS
from emotional_instability.training.train_dpo import train_dpo
from emotional_instability.training.train_sft import train_sft


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    p = base_parser(__doc__)
    p.add_argument("--method", choices=["dpo", "sft"], required=True)
    p.add_argument("--ablation", choices=list(ABLATION_BANDS), default=None,
                   help="DPO only: restrict LoRA to a layer band (Appendix I)")
    args = p.parse_args()
    cfg = config_from_args(args)

    if args.method == "dpo":
        pairs = _load_json(os.path.join(cfg.output_dir, "training", "dpo", "pairs.json"))
        layers = ABLATION_BANDS.get(args.ablation) if args.ablation else None
        subdir = f"dpo_{args.ablation}" if args.ablation else "dpo"
        adapter = train_dpo(cfg, pairs, layers=layers, output_subdir=subdir)
    else:
        dataset = _load_json(os.path.join(cfg.output_dir, "training", "sft", "dataset.json"))
        adapter = train_sft(cfg, dataset)

    print(f"Adapter saved to: {adapter}")


if __name__ == "__main__":
    main()
