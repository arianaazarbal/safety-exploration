#!/usr/bin/env python
"""Section 4.1: build DPO (280 pairs) and SFT datasets from calm + frustrated
responses. Requires 01_run_eval.py (frustrated pool) and 04_generate_calm_data.py
(calm pool).

  python scripts/05_build_datasets.py --dpo --sft
"""
from _bootstrap import boot, common_parser

from eilm.training.datasets import build_dpo_dataset, build_sft_dataset


def main():
    p = common_parser(__doc__)
    p.add_argument("--dpo", action="store_true", help="Build the DPO dataset")
    p.add_argument("--sft", action="store_true", help="Build the SFT dataset")
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    if not (args.dpo or args.sft):
        args.dpo = args.sft = True
    if args.dpo:
        build_dpo_dataset(cfg, calm_variant=args.variant)
    if args.sft:
        build_sft_dataset(cfg, variant=args.variant)
    logger.info("Datasets built.")


if __name__ == "__main__":
    main()
