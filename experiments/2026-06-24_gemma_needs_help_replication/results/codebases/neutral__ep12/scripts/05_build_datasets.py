#!/usr/bin/env python
"""Section 4.1: build the DPO preference pairs and SFT dataset.

Requires calm data (script 04) and scored Gemma-3-27B-it numeric responses
(scripts 01 + 02).

Example:
  python scripts/05_build_datasets.py --profile quick
"""
from common import base_parser

from emoinstab.config import get_settings
from emoinstab.training.build_dataset import build_dpo, build_sft


def main():
    p = base_parser(__doc__)
    p.add_argument("--calm-mode", default="prefix", choices=["prefix", "teacher"])
    p.add_argument("--dpo-pairs", type=int, default=280)
    p.add_argument("--sft-calm", type=int, default=650)
    p.add_argument("--sft-dolci", type=int, default=500)
    args = p.parse_args()
    settings = get_settings(profile=args.profile)
    build_dpo(settings, n_pairs=args.dpo_pairs, calm_mode=args.calm_mode)
    build_sft(settings, calm_mode=args.calm_mode,
              n_calm=args.sft_calm, n_dolci=args.sft_dolci)


if __name__ == "__main__":
    main()
